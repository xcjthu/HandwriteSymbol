import json
import csv
path = "/data/disk1/private/xcj/BigDataClass/result.json"
result = json.load(open(path, "r"))

headers = ["user_id", "article_1", "article_2", "article_3", "article_4", "article_5"]
rows = []
for u in result:
    cand = list(zip(u[0], u[1]))
    cand.sort(key = lambda x:x[0], reverse = True)
    out = [c[1] for c in cand[:5]]
    uid = u[2]
    row = [uid, out[0], out[1], out[2], out[3], out[4]]
    rows.append(row)

with open('submit.csv','w') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(headers)
    f_csv.writerows(rows)

