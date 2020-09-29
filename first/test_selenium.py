from selenium import webdriver
import time
from selenium.webdriver.support.select import Select





driver = webdriver.Chrome("E:\\setup\\chromedriver.exe")
driver.implicitly_wait(3)
driver.maximize_window()
driver.get("http://127.0.0.1:8000/")

driver.find_element_by_name("c1").click()
time.sleep(2)
driver.find_element_by_name("c2").click()
time.sleep(2)
driver.find_element_by_id("elbowbtn").click()
driver.find_element_by_id("knnbtn").click()
time.sleep(2)
driver.find_element_by_name("c3").click()
time.sleep(2)

driver.find_element_by_id("nbcbtn").click()
time.sleep(2)
driver.find_element_by_id("nbcpredict").click()
driver.get("http://127.0.0.1:8000/predict_price_nbc/")
time.sleep(1)
driver.find_element_by_id("CLOCK_SPEED").send_keys(3)
time.sleep(1)
driver.find_element_by_id("FRONT_CAMERA").send_keys(20)
time.sleep(1)
driver.find_element_by_id("PRIMARY_CAMERA").send_keys(40)
time.sleep(1)
driver.find_element_by_id("INTERNAL_MEMORY").send_keys(64)
time.sleep(1)
driver.find_element_by_id("NUMBER_OF_CORES").send_keys(6)
time.sleep(1)
driver.find_element_by_id("RAM").send_keys(5)
time.sleep(1)
driver.find_element_by_id("submitForm").click()
time.sleep(2)


driver.get("http://127.0.0.1:8000/")

driver.find_element_by_name("c4").click()
time.sleep(2)

driver.find_element_by_id("lrbtn").click()
time.sleep(2)
driver.find_element_by_id("lrpredict").click()
driver.get("http://127.0.0.1:8000/predict_price_lr/")
time.sleep(1)
driver.find_element_by_id("CLOCK_SPEED").send_keys(3)
time.sleep(1)
driver.find_element_by_id("FRONT_CAMERA").send_keys(20)
time.sleep(1)
driver.find_element_by_id("PRIMARY_CAMERA").send_keys(40)
time.sleep(1)
driver.find_element_by_id("INTERNAL_MEMORY").send_keys(64)
time.sleep(1)
driver.find_element_by_id("NUMBER_OF_CORES").send_keys(6)
time.sleep(1)
driver.find_element_by_id("RAM").send_keys(5)
time.sleep(1)
driver.find_element_by_id("submitForm").click()
time.sleep(2)

driver.close()

driver.quit()
