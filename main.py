import argparse
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BERTClass(nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.roberta = AutoModel.from_pretrained("models/rubert-tiny2")
        self.fc = nn.Linear(312, 248)

    def forward(self, ids, mask, token_type_ids):
        _, features = self.roberta(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output = self.fc(features)
        return output


class BERTDataset(Dataset):
    def __init__(self, X, tokenizer, max_len):
        self.len = len(X)
        self.X = X.reset_index(drop=True)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        text = self.X.iloc[index]
        inputs = self.tokenizer.encode_plus(
            text,
            truncation=True,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }

def classificatoin(X_test):
    # Impute nans with empty string for simplicity
    imputer = SimpleImputer(fill_value='', strategy='constant')
    X_test = pd.DataFrame(imputer.fit_transform(X_test), columns = X_test.columns)

    # Concatenate the features
    X_test_concat = X_test.demands + ' ' + X_test.company_name + ' ' +  X_test.achievements_modified

    # Load the encoder
    oe = pickle.load(open('models/classification/ordinal_encoder.pkl', 'rb'))
    encoder = pickle.load(open('models/classification/one_hot_encoder.pkl', 'rb'))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/rubert-tiny2")

    # Load the model
    model = BERTClass()
    model.load_state_dict(torch.load('models/classification/job_name_model.pt'))
    model.to(device)

    # Create the dataloader
    valid_dataset = BERTDataset(X_test_concat, tokenizer, 256)
    valid_loader = DataLoader(valid_dataset, batch_size=64, num_workers=4, shuffle=False, pin_memory=True)

    # Compute the model predictions
    model.eval()
    y_preds = []
    with torch.no_grad():
        for _, data in enumerate(valid_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            _, preds = torch.max(outputs, dim = 1)
            y_preds.extend(preds)
    y_preds_processed = np.array([])

    for tens in y_preds:
        tens = tens.cpu().numpy()
        y_preds_processed = np.append(y_preds_processed, tens)

    # Create the dataframe with the predictions
    classification_result_df = pd.DataFrame(columns = ['id', 'job_name', 'task_type'])
    classification_result_df['id'] = X_test['id']
    classification_result_df['job_name'] = oe.inverse_transform(y_preds_processed.reshape(-1, 1)).ravel()
    classification_result_df['task_type'] = 'RES'

    return classification_result_df


def regression(X_test):
    pass


def main():
    # parse arguments
    parser = argparse.ArgumentParser(description='My CLI tool')
    parser.add_argument('-job', type=str, required=True, help='path to job test .csv file')
    parser.add_argument('-sal', type=str, required=True, help='path to sallary test .csv file')
    parser.add_argument('-sub', type=str, required=True, help='where to save submission .csv file')

    args = parser.parse_args()
    job_test = pd.read_csv(args.job)
    sal_test = pd.read_csv(args.sal)

    # Compute the first task
    job_result = classificatoin(job_test)

    # Compute the second task
    sal_result = regression(sal_test)

    # Save the result
    submission = pd.concat([job_result, sal_result], axis=0)
    submission.to_csv(args.sub, index = False)


if __name__ == "__main__":
    main()