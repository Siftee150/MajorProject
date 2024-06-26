from decimal import Decimal
import urllib.request as urlrequest
import json
import pandas as pd
import os


http_headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
                ' Chrome/58.0.3029.110 Safari/537.36'}


def wei2ether(s):
    length = len(s)
    t = length - 18
    if (t > 0):
        s1 = ""
        s1 = s1+s[0:t]
        s1 = s1+"."
        s1 = s1+s[t:]
    else:
        x = 18-length
        s1 = "0."
        for i in range(0, x):
            s1 = s1+"0"
        s1 = s1+s
    # print(s+" "+s1)
    return Decimal(s1)


def load_Tx(list_result):
    df_out = pd.DataFrame(columns=['TxHash', 'BlockHeight', 'TimeStamp', 'From', 'To', 'Value',
                                   'ContractAddress', 'Input', 'isError'])
    for dic_txs in list_result:
        t1 = (dic_txs['hash'], dic_txs['blockNumber'], dic_txs['timeStamp'], dic_txs["from"], dic_txs["to"],
              wei2ether(dic_txs["value"]),
              dic_txs["contractAddress"], dic_txs["input"], dic_txs["isError"])
        t = []
        for x in t1:
            if x == "":
                x = 'NULL'
            t.append(x)
        s = pd.DataFrame(
            {'TxHash': t[0], 'BlockHeight': t[1], 'TimeStamp': t[2], 'From': t[3], 'To': t[4],
             'Value': t[5], 'ContractAddress': t[6], 'Input': t[7], 'isError': t[8]}, index=[0])
        df_out = pd.concat([df_out, s], axis=0, ignore_index=True)
    return df_out


def load_url(phishing_address, saved_address):
    if phishing_address[0] != '0':
        print("not exist")
        return False
    elif phishing_address == "not exist":
        print("not exist")
        return False
    else:
        url_outer = 'http://api.etherscan.io/api?module=account&action=txlist&' \
                    'address=' + phishing_address + '&startblock=0&endblock=99999999&sort=asc&' \
                                                    'apikey=' + \
            config["API_KEY"]
        crawl_outer = urlrequest.urlopen(url_outer).read()
        json_outer = json.loads(crawl_outer.decode('utf8'))
        if json_outer["status"] == "1":
            result_outer = json_outer['result']
        elif json_outer["status"] == "0":
            result_outer = []
        df_outer = load_Tx(result_outer)
        url_inter = 'http://api.etherscan.io/api?module=account&action=txlistinternal&address=' \
                    + phishing_address + '&startblock=0&endblock=99999999&sort=asc&apikey=' \
                    + config['API_KEY']
        crawl_inter = urlrequest.urlopen(url_inter).read()
        json_inter = json.loads(crawl_inter.decode('utf8'))
        if json_inter["status"] == "1":
            result_inter = json_inter['result']
        elif json_inter["status"] == "0":
            result_inter = []
        df_inter = load_Tx(result_inter)
        df_outer = pd.concat([df_outer, df_inter], axis=0, ignore_index=True)
        df_outer = df_outer.sort_values(by="TimeStamp")
        df_outer.to_csv(saved_address+'/'+phishing_address + '.csv')
        return True


def get_neighbor_list(prev, address):
    df_address = pd.read_csv('./'+prev+'/'+address+'.csv')
    if len(df_address.index) >= 10000:
        return []
    # print(df_address)
    set_neighbor = set()
    newpath = prev + '/' + address + '/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i in df_address.index:
        # print(df_address.Value[i])
        if df_address.Value[i] != 0:
            if df_address.isError[i] == 0:
                if str.lower(df_address.From[i]) != str.lower(address):
                    set_neighbor.add(str.lower(df_address.From[i]))
                elif str.lower(df_address.To[i]) != str.lower(address):
                    set_neighbor.add(str.lower(df_address.To[i]))

    list_neighbor = list(set_neighbor)
    df_neighbor = pd.DataFrame(data=list_neighbor, columns=['address'])
    df_neighbor.to_csv('./'+prev+'/'+address+'_neighbor.csv')
    return list_neighbor


def get_k_order_neighbor(k, i, prev, address):
    if k == i-1:
        return
    # print(i, '   order')
    status = load_url(address, prev)
    if (status == True):
        list_neighbor = get_neighbor_list(prev, address)
        # print(len(list_neighbor))
        for single in list_neighbor:
            get_k_order_neighbor(k, i+1, prev+'/'+address, single)


def read_data(k, filename, b, e=1):
    df_address = pd.read_csv(filename, names=['address'])
    # print(df_address.size)
    for i in range(b, e):
        address = df_address.address[i]
        # print('------------------------------------')
        # print('begin', i, address)
        newpath = config['DATASET_PATH']+'/'+address + '_data'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        get_k_order_neighbor(k, 0, newpath, address)
        # print('------------------------------------')


with open('config.json', 'r') as f:
    config = json.load(f)

read_data(1, 'combinedaddresses.csv', 0, 5)
