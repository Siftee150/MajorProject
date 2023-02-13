import pickle
from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FireFoxService
from webdriver_manager.firefox import GeckoDriverManager
import json
from selenium.webdriver.common.by import By
from urllib.parse import urlparse
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

URL_LOGIN = 'https://etherscan.io/login'
# Login to etherscan and auto fill login information if available

address_list = []
num_pages = 0


def login():
    driver.get(URL_LOGIN)
    driver.implicitly_wait(5)
    driver.find_element("id",
                        "ContentPlaceHolder1_txtUserName").send_keys(config['ETHERSCAN_USER'])
    driver.find_element(
        "id", "ContentPlaceHolder1_txtPassword").send_keys(config['ETHERSCAN_PASS'])

    input("Press enter once logged in")


def getphishingAccounts():
    driver.get(
        'https://etherscan.io/tokens/label/phish-hack?subcatid=0&size=100&start=0&col=3&order=desc')
    num_pages = WebDriverWait(driver, 20).until(EC.visibility_of_element_located(
        (By.XPATH, "/html/body/main/section[2]/div[1]/div[3]/div/div/div/div/div[1]/div[2]/ul/li[3]/span/span[2]"))).text
    c = 1
    start = 0
    while (c <= int(num_pages)):
        links = driver.find_elements(
            By.XPATH, '//div[@class="d-flex align-items-center gap-1"]/a')
        for link in links:
            p = urlparse(link.get_attribute('href'))
            if all([p.scheme, p.netloc]):
                address_list.append(p.path.rsplit("/", 1)[-1])
        c = c+1
        start = start+100
        driver.get(
            'https://etherscan.io/tokens/label/phish-hack?subcatid=0&size=100&start='+str(start)+'&col=3&order=desc')
    print(len(address_list))
    with open('etherscanfile.pkl', 'wb') as file:
        pickle.dump(address_list, file)


driver = webdriver.Firefox(service=FireFoxService(
    GeckoDriverManager().install()))

with open('config.json', 'r') as f:
    config = json.load(f)
login()
getphishingAccounts()
