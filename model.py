import torch
import torch.nn as nn
import torch.nn.functional as F

def log_Bernoulli(x, mean, average=False, dim=1):
    """
    Bernoulli log-likelihood
    """
    min_epsilon = 1e-5
    max_epsilon = 1. - min_epsilon
    probs = torch.clamp(mean, min=min_epsilon, max=max_epsilon)
    log_bernoulli = x * torch.log(probs) + (1 - x) * torch.log(1 - probs)
    if average:
        return torch.mean(log_bernoulli, dim)
    else:
        return torch.sum(log_bernoulli, dim)


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.L = args.L
        self.D = args.D
        self.K = args.K

        self.self_attention_mode = args.self_att
        self.kernel_self_attention = args.kernel_self_att
        self.init = args.init

        self.classification_threshold = 0.5

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 50, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        if self.init:
            torch.nn.init.xavier_uniform_(self.feature_extractor[0].weight)
            self.feature_extractor[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.feature_extractor[3].weight)
            self.feature_extractor[3].bias.data.zero_()

        self.fc = nn.Sequential(
            nn.Linear(50 * 4 * 4, self.L),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.L, self.L),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        if self.init:
            torch.nn.init.xavier_uniform_(self.fc[0].weight)
            self.fc[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.fc[3].weight)
            self.fc[3].bias.data.zero_()

        if self.self_attention_mode:
            self.self_att = SelfAttention(self.L, self.kernel_self_attention)

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K) # outputs N x K
        )

        if self.init:
            torch.nn.init.xavier_uniform_(self.attention[0].weight)
            self.attention[0].bias.data.zero_()
            torch.nn.init.xavier_uniform_(self.attention[2].weight)
            self.attention[2].bias.data.zero_()

        self.classifier = nn.Sequential(
            nn.Linear(self.L * self.K, 1),
            nn.Sigmoid()
        )

        if self.init:
            torch.nn.init.xavier_uniform_(self.classifier[0].weight)
            self.classifier[0].bias.data.zero_()

    def forward(self, x):
        x = x.squeeze(0) # maybe not necessary?

        H = self.feature_extractor(x)
        H = H.view(-1, 50 * 4 * 4)
        H = self.fc(H)

        gamma, gamma_kernel = (0, 0)
        if self.self_attention_mode:
            H, self_attention, gamma, gamma_kernel = self.self_att(H)

        # attention
        A = self.attention(H)  # outputs N x K
        A = torch.transpose(A, 1, 0)  # shape K x N,
        z = F.softmax(A, dim=1)  # softmax w/o dim=1 takes last dim. by default; weights a_i

        M = torch.mm(z, H)  # shape (K x N).(N x L) -> K x L: attention output operator
        M = M.view(1, -1)

        y_prob = self.classifier(M)
        y_hat = torch.ge(y_prob, self.classification_threshold).float()

        if self.self_attention_mode:
            return y_prob, y_hat, z, (A, self_attention), gamma, gamma_kernel
        else:
            return y_prob, y_hat, z, A, gamma, gamma_kernel

    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, y_hat, _, _, gamma, gamma_kernel = self.forward(X)
        error = 1.0 - y_hat.eq(Y).cpu().float().mean()
        return error, gamma, gamma_kernel

    def calculate_objective(self, X, Y):
        Y = Y.float()
        y_prob, _, _, _, gamma, gamma_kernel = self.forward(X)
        log_likelihood = -log_Bernoulli(Y, y_prob)
        return log_likelihood, gamma, gamma_kernel

    def calculate_prediction(self, X, Y):
        Y = Y.float()
        y_prob, y_hat, _, _, _, _ = self.forward(X)
        return y_prob, y_hat


class SelfAttention(nn.Module):
    def __init__(self, in_dim, kernel=False):
        super(SelfAttention, self).__init__()
        self.kernel_self_attention = kernel
        self.query_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))  # in the article it is initialized to 0!
        self.softmax = nn.Softmax(dim=-1)
        self.gamma_att = nn.Parameter(torch.ones(1))  # alpha from article

    def forward(self, x):
        x = x.view(1, x.shape[0], x.shape[1]).permute((0, 2, 1))
        bs, C, length = x.shape
        proj_query = self.query_conv(x).view(bs, -1, length).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(bs, -1, length)

        if self.kernel_self_attention:
            proj = torch.zeros((length, length))
            proj_query = proj_query.permute(0, 2, 1)
            for i in range(length):
                gauss = torch.pow(proj_query - proj_key[:, :, i].t(), 2).sum(dim=1) # Q_i - K_j as defined in Kim et al.
                proj[:, i] = torch.exp(-F.relu(self.gamma_att) * gauss)
            energy = proj.view((1, length, length))
        else:
            energy = torch.bmm(proj_query, proj_key)

        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(bs, -1, length)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(bs, C, length)

        out = self.gamma * out + x

        return out[0].permute(1, 0), attention, self.gamma, self.gamma_att
