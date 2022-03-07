from ipaddress import ip_address
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoConfig, AutoModelForSequenceClassification


model_ckpt = "URLTran-BERT"
config = AutoConfig.from_pretrained(model_ckpt)
config.num_labels = 2
config.problem_type = 'single_label_classification'

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class URLDatasetClassification(Dataset):
    def __init__(self, encodings, labels):
        super(URLDatasetClassification).__init__()
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        X = self.encodings.input_ids[idx]
        y = self.labels[idx]
        return X, y

    def __len__(self):
        return len(self.encodings.input_ids)


def data_prep(dataset_path):
    df_final = pd.read_csv(dataset_path)
    X = df_final.url
    y = df_final.label.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    X_test, X_eval, y_test, y_eval = train_test_split(X_test, y_test, test_size=0.10, random_state=42)

    train_df = pd.DataFrame(zip(X_train.values, y_train.values), columns=['url','label'])
    test_df = pd.DataFrame(zip(X_test.values, y_test.values), columns=['url','label'])
    eval_df = pd.DataFrame(zip(X_eval.values, y_eval.values), columns=['url','label'])
    return train_df, eval_df


def preprocess(url_data, tokenizer):
    inputs = tokenizer(
        url_data,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )

    return inputs


def predict(url, tokenizer, model):
    inputs = preprocess(url, tokenizer)
    return torch.argmax(torch.softmax(model(**inputs).logits, dim=1)).tolist()


def train_model(train_df, tokenizer, model):
    train_url_data = train_df.url.values.tolist()
    train_url_labels = train_df.label.values.tolist()
    train_inputs = preprocess(train_url_data, tokenizer)

    train_dataset = URLDatasetClassification(train_inputs, train_url_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # model training
    model.to(device)
    model.train()

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    epochs = 10
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            # prep data for predict step
            inputs, labels = batch
            X = inputs.to(device)
            y = labels.to(device)

            outputs = model(X, labels=y)
            import ipdb; ipdb.set_trace()
            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {loss.item()}")
        model.save_pretrained(f"URLTran-BERT-CLS-{epoch}")


def eval_model(eval_df, tokenizer, model):
    eval_url_data = eval_df.url.values.tolist()
    eval_url_labels = eval_df.label.values.tolist()
    eval_inputs = preprocess(eval_url_data, tokenizer)

    eval_dataset = URLDatasetClassification(eval_inputs, eval_url_labels)
    eval_loader = DataLoader(eval_dataset, batch_size=2000, shuffle=True)

    y_true = []
    y_pred = []

    model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            inputs, labels = batch
            X_eval = inputs.to(device)
            y_eval = labels.to(device)

            outputs = model(X_eval, labels=y_eval)
            predictions = [torch.argmax(pred).tolist() for pred in torch.softmax(outputs.logits, dim=1)]
            y_eval_true = y_eval.tolist()

            y_true.extend(y_eval_true)
            y_pred.extend(predictions)

        total_acc = accuracy_score(y_true, y_pred)
        total_f1 = f1_score(y_true, y_pred)
        print(f"Acc: {total_acc} F1: {total_f1}")


if __name__ == "__main__":
    train_df, eval_df = data_prep("data/final_data.csv")
    train_model(train_df, tokenizer, model)
    eval_model(eval_df, tokenizer, model)