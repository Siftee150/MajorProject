import pickle
import pandas as pd
import csv
with open('etherscanfile.pkl', 'rb') as f:
    etherscan_addresses = pickle.load(f)
print("Total etherscan addresses are "+str(len(etherscan_addresses)))
etherscan_addresses = [x.lower() for x in etherscan_addresses]
# print(etherscan_addresses)
cryptoscam_df = pd.read_csv('phishingaddress.csv', names=['address'])
cryptoscam_addresses = cryptoscam_df['address'].tolist()
print("Total cryptoscam addresses are "+str(len(cryptoscam_addresses)))
cryptoscam_addresses = [x.lower() for x in cryptoscam_addresses]
total_addresses = list(set().union(cryptoscam_addresses, etherscan_addresses))
print("Total combined addresses are "+str(len(total_addresses)))

# print(total_addresses)

file = open('combinedaddresses.csv', 'w+', newline='')

with file:
    writer = csv.writer(file, delimiter=",")
    writer.writerows([c.strip() for c in r.strip(', ').split(',')]
                     for r in total_addresses)
