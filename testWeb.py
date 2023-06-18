import unittest
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.chromium import service
from selenium.common.exceptions import NoSuchElementException


class testWeb(unittest.TestCase):

	def setUp(self):
		service = Service(
							executable_path='http://127.0.0.1:5000/')
		self.driver = webdriver.Chrome(service=service)
		self.driver.get("http://127.0.0.1:5000/")

	# def test_prediction(self):
	# 	try:
	# 		input_element = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.NAME, "sentiment")))
	# 		input_element.send_keys("1")
	# 		button = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.ID, "button-predict")))
	# 		button.click()
	# 		WebDriverWait(self.driver, 20).until(EC.visibility_of_element_located((By.ID, "prediction_result"))).text
	# 	except NoSuchElementException:
	# 		print('Predictionion Test Failed')
			
	# 	print('Predictionion Test Done ✓')
	# 	time.sleep(3)


	# def test_view_dataset(self):
	# 	try:
	# 		navData = WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.ID, "data")))
	# 		navData.click()
	# 		WebDriverWait(self.driver, 10).until(EC.visibility_of_element_located((By.CLASS_NAME, "dataframe")))
	# 	except NoSuchElementException:
	# 		print('View Dataset Test Failed')

	# 	print('View Dataset Test Done ✓')
	# 	time.sleep(3)

	# def test_preprocessing(self):
	# 	try:
	# 		navSimulation = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
	# 		navSimulation.click()
	# 		navPreprocessing = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "preprocessing")))
	# 		navPreprocessing.click()
	# 		buttonPreprocessing = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "button-preprocessing")))
	# 		buttonPreprocessing.click()
	# 		WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.CLASS_NAME, "dataframe")))
	# 	except NoSuchElementException:
	# 		print('Preprocessing Test Failed')

	# 	print('Preprocessing Test Done ✓')
	# 	time.sleep(3)

	# def test_weighting(self):
	# 	try:
	# 		navSimulation = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
	# 		navSimulation.click()
	# 		navTfidf = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "weighting")))
	# 		navTfidf.click()
	# 		buttonSplit = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "button-split")))
	# 		buttonSplit.click()
	# 		WebDriverWait(self.driver, 20).until(
	# 			EC.visibility_of_element_located((By.ID, "train-data"))).text
	# 		WebDriverWait(self.driver, 20).until(
	# 			EC.visibility_of_element_located((By.ID, "test-data"))).text
	# 		buttonWeighting = WebDriverWait(self.driver, 10).until(
	# 			EC.visibility_of_element_located((By.ID, "button-weighting")))
	# 		buttonWeighting.click()
	# 		WebDriverWait(self.driver, 20).until(
	# 			EC.visibility_of_element_located((By.ID, "train-shape"))).text
	# 		WebDriverWait(self.driver, 20).until(
	# 			EC.visibility_of_element_located((By.ID, "test-shape"))).text

			
	# 	except NoSuchElementException:
	# 		print('Weighting Using TF-IDF Test Failed')

	# 	print('Weighting Using TF-IDF Test Done ✓')
	# 	time.sleep(3)

	def test_svm(self):
		try:

			#Do preprocesing
			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navPreprocessing = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "preprocessing")))
			navPreprocessing.click()
			buttonPreprocessing = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-preprocessing")))
			buttonPreprocessing.click()

			#Do Weighitng
			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navTfidf = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "weighting")))
			navTfidf.click()
			buttonSplit = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-split")))
			buttonSplit.click()
			buttonWeighting = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-weighting")))
			buttonWeighting.click()

			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navSvm = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "svm")))
			navSvm.click()
			buttonSvm = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-svm")))
			buttonSvm.click()
			WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.CLASS_NAME, "accuracy_result_text")))
			WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.CLASS_NAME, "image_plot_svm")))
			WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.CLASS_NAME, "dataframe")))
			
		except NoSuchElementException:
			print('SVM Process Test Failed')

	print('SVM Process Test Done ✓')
	time.sleep(3)

	def test_ga(self):
		try:
			# Do preprocesing
			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navPreprocessing = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "preprocessing")))
			navPreprocessing.click()
			buttonPreprocessing = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-preprocessing")))
			buttonPreprocessing.click()

			# Do Weighitng
			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navTfidf = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "weighting")))
			navTfidf.click()
			buttonSplit = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-split")))
			buttonSplit.click()
			buttonWeighting = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "button-weighting")))
			buttonWeighting.click()

			navSimulation = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "navbarDropdownMenuLink")))
			navSimulation.click()
			navGa = WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.ID, "ga")))
			navGa.click()

			# # Fill the form
			WebDriverWait(self.driver, 100).until(
				EC.visibility_of_element_located((By.NAME, "population"))).send_keys(10)
			WebDriverWait(self.driver, 100).until(
				EC.visibility_of_element_located((By.NAME, "crossover"))).send_keys(float(0.09))
			WebDriverWait(self.driver, 100).until(
				EC.visibility_of_element_located((By.NAME, "mutation"))).send_keys(float(0.09))
			WebDriverWait(self.driver, 100).until(
				EC.visibility_of_element_located((By.NAME, "generation"))).send_keys(10)

			buttonGa = WebDriverWait(self.driver, 10).until(
											EC.visibility_of_element_located((By.ID, "button-ga")))
			buttonGa.click()
			WebDriverWait(self.driver, 10).until(
				EC.visibility_of_element_located((By.CLASS_NAME, "image_plot_ga")))


		except NoSuchElementException:
			print('Genetic Algorithm Test Failed')

		print('Genetic Algorithm Test Done ✓')
		time.sleep(3)
	
	


def tearDown(self):
	self.driver.close()


if __name__ == "__main__":
    unittest.main()
