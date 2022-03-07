import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, BertForMaskedLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# import and preprocess data
df_final = pd.read_csv("data/final_data.csv")
url_data = df_final.url.values[:10000].tolist()


class URLDataset(Dataset):
    def __init__(self, encodings):
        super(URLDataset).__init__()
        self.encodings = encodings

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def preprocess(url_data, tokenizer):
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
    rand = torch.rand(inputs.shape)
    # mask array that replicates BERT approach for MLM
    # ensure that [cls], [sep], [mask] remain untouched
    mask_arr = (
        (rand < 0.15)
        * (inputs != 101)
        * (inputs != 102)
        * (inputs != 0)
    )

    selection = [
        torch.flatten(mask_arr[i].nonzero()).tolist()
        for i in range(inputs.shape[0])
    ]

    for i in range(inputs.shape[0]):
        inputs[i, selection[i]] = 103

    return inputs


def predict_mask(url, tokenizer, model):
    inputs = preprocess(url, tokenizer)
    masked_inputs = masking_step(inputs["input_ids"]).to(device)
    with torch.no_grad():
        predictions = model(masked_inputs)

    output_ids = torch.argmax(
        torch.nn.functional.softmax(predictions.logits[0], -1),dim=1).tolist()

    return masked_inputs, output_ids


def train(url_data, tokenizer, model):
    inputs = preprocess(url_data, tokenizer)

    # stage data for Pytorch
    dataset = URLDataset(inputs)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # model training
    model.to(device)
    model.train()

    # initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    epochs = 2
    for epoch in range(epochs):
        for batch in loader:
            optimizer.zero_grad()

            # prep data for predict step
            masked_inputs = masking_step(batch["input_ids"]).to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(masked_inputs, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} Loss: {loss.item()}")
        model.save_pretrained(f"URLTran-BERT-{epoch}")


if __name__ == "__main__":
    train(url_data, tokenizer, model)
    url = "huggingface.co/docs/transformers/task_summary"
    input_ids, output_ids = predict_mask(url, tokenizer, model)

    masked_input = "".join([tokenizer.ids_to_tokens[tok_id].replace("##", "") for tok_id in input_ids[0].tolist()])
    prediction = "".join([tokenizer.ids_to_tokens[tok_id].replace("##", "") for tok_id in output_ids])

    print(f"Masked Input: {masked_input}")
    print(f"Predicted Output: {prediction}")

