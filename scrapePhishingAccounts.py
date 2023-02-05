from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
import pickle
http_headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
                ' Chrome/58.0.3029.110 Safari/537.36'}
URL = "https://cryptoscamdb.org/scams"
driver = webdriver.Firefox()
driver.get(url=URL)
driver.maximize_window()  # For maximizing window
driver.implicitly_wait(20)  # gives an implicit wait for 20 seconds
# pointer starting from 2 as

url_list = []
reached_end = False
while True:
    c = 2
    while True:
        try:
            baseRow = '//html/body/div/div/div/div[3]/div[1]/div/table/tbody/tr['+str(
                c)+']'
            categoryXPath = baseRow+'/td[4]'
            category = driver.find_element(By.XPATH, categoryXPath)
            if (category.text == 'Phishing'):
                accountUrlXpath = baseRow+'/td[2]/a'
                accountUrl = driver.find_element(By.XPATH, accountUrlXpath)
                link = accountUrl.get_attribute('href')
                url_list.append(link)
        except:
            if (c == 2):
                reached_end = True
            break
        c = c+1
    if (reached_end == True):
        break
    try:
        pageXpath = '/html/body/div/div/div/div[3]/div[1]/div/div/div/div[6]'
        page = driver.find_element(By.XPATH, pageXpath)
        page.click()
    except:
        break
print(url_list)
with open('file.pkl', 'wb') as file:
    # A new file will be created
    pickle.dump(url_list, file)
