import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForMaskedLM


class URLDataset(Dataset):
    def __init__(self, encodings):
        super().__init__()
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def preprocess(url_data):
    inputs = tokenizer(
        url_data,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True
    )

    inputs["labels"] = inputs.input_ids.detach().clone()
    return inputs


def masking_step(inputs):
    rand = torch.rand(inputs.input_ids.shape)
    # mask array that replicates BERT approach for MLM
    mask_arr = (
        (rand < 0.15)
        * (inputs.input_ids != 101)
        * (inputs.input_ids != 102)
        * (inputs.input_ids != 0)
    )

    selection = [
        torch.flatten(mask_arr[i].nonzero()).tolist()
        for i in range(inputs.input_ids.shape[0])
    ]

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    return inputs


def predict_mask(url):
    inputs = preprocess(url)
    inputs = masking_step(inputs)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    labels = inputs["labels"].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

    output_ids = [
        torch.argmax(torch.nn.functional.softmax(outputs.logits[0][i], dim=0)).item()
        for i in range(outputs.logits[0].shape[0])
    ]
    return input_ids, output_ids

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# import and preprocess data
df_final = pd.read_csv("final_data.csv")
url_data = df_final.url.values.tolist()
inputs = preprocess(url_data)
inputs = masking_step(inputs)

# stage data for Pytorch
dataset = URLDataset(inputs)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# model training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.train()

# initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

epochs = 2
for epoch in range(epochs):
    for batch in loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print("Epoch: {} Loss: {}".format(epoch, loss.item()))

url = "huggingface.co/docs/transformers/task_summary"
input_ids, output_ids = predict_mask(url)

print("Masked Input: {}".format("".join(
    [tokenizer.ids_to_tokens[tok_id].replace("##", "") for tok_id in input_ids[0].tolist()])))

print("Predicted Output: {}".format("".join(
    [tokenizer.ids_to_tokens[tok_id].replace("##", "") for tok_id in output_ids])))
