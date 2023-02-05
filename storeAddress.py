import pickle
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
driver = webdriver.Firefox()
file = open('./file.pkl', 'rb')
url_data = pickle.load(file)
print("Parsing total urls "+str(len(url_data)))
phishing_addresses = []
for url in url_data:
    addressXpath = "/html/body/div/div/div/div[3]/div[1]/div[1]/div[8]/ul/li/a"
    try:
        driver.get(url=url)
        address = driver.find_element(By.XPATH, addressXpath)
        phishing_addresses.append([address.text])
    except:
        continue
with open('phishingaddress', 'w') as f:
    write = csv.writer(f)
    write.writerows(phishing_addresses)
file.close()
