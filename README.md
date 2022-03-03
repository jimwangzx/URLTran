# URLTran
PyTorch implementation of Improving Phishing URL Detection via Transformers [Paper](https://arxiv.org/pdf/2106.05256.pdf)

## Data
The paper used ~1.8M URLs (90/10 split on benign vs. malicious). There are few places to gather malicious URLs. My recommendation is to do the following:

### Malicious URLs
__OpenPhish__ will provide 500 malicious URLs for free in TXT form. You can access that data [here](https://openphish.com/phishing_database.html).

Likewise, __PhishTank__ is an excellent resource that provides a daily feed of malicious URLs in CSV or JSON format. You can gather ~5K through the following [link](https://www.phishtank.com/developer_info.php).

Finally, there is an excellent OpenSource project, [Phishing.Database](https://github.com/mitchellkrogza/Phishing.Database), run by Mitchell Krog. There is a ton of data available here to plus up your dataset.

### Benign Data
I gathered benign URL data via two methods. The first was to use the top 50K domains from [Alexa](http://s3.amazonaws.com/alexa-static/top-1m.csv.zip).

Next I used my own Chrome browser history to get an additional 60K. It was pretty easy to do on my Macbook. First, make sure your browser is closed. Then in your terminal run the following command:

```bash
/usr/bin/sqlite3 -csv -header ~/Library/Application\ Support/Google/Chrome/Default/History "SELECT urls.id, urls.url FROM urls JOIN visits ON urls.id = visits.url LEFT JOIN visit_source ON visits.id = visit_source.id order by last_visit_time asc;" > history.csv
```

## Tasks
### Masked Language Modeling

```python
# Input:
    url = "huggingface.co/docs/transformers/task_summary"
    input_ids, output_ids = predict_mask(url, tokenizer, model)

# Output:
    Masked Input: [CLS]huggingface.co[MASK]docs[MASK]transformers/task_summary[SEP]
    Predicted Output: [CLS]huggingface.co/docs/transformers/task_summary[SEP]
```