import csv

csv_file = open('parsed.csv', mode='w')
fieldnames = ['original', 'edited', 'meanGrade']
writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
writer.writeheader()

with open("train_funlines.csv", newline='') as f:
    spamreader = csv.DictReader(f)
    for row in spamreader:
        temp =row['original']
        start = temp.index("<")
        end = temp.index(">")
        edited = temp[0:start] + row['edit'] + temp[end+1:]
        original = temp[0:start] + temp[start+1:temp.index("/")] + temp[end+1:]
        writer.writerow({'original': original, 'edited': edited, 'meanGrade': row["meanGrade"]})

