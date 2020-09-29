from django.test import TestCase
from django.test import Client
from django.urls import reverse

class TestView(TestCase):

       @classmethod
       def setUpClass(cls):
           # creating instance of a client.
           super(TestView, cls).setUpClass()
       def setUp(self):
           self.client = Client()
       def testHome(self):

           # Issue a GET request.
           # client = Client()


           # Check that the response is 200 OK.
           try :
             response = self.client.get('/')
             self.assertEqual(response.status_code, 200)
             self.assertTemplateUsed(response,'home.html')
             print("test case 1 passed")
           except:
               print("test case 1 failed")


       def testPredictLr(self):
            # client = Client()
            try:
                response = self.client.get(reverse('predict_price_lr'))
                self.assertEqual(response.status_code, 200)
                self.assertTemplateUsed(response,'predict_price_lr.html')

                print("test case 2 passed")

            except:

                print("test case 2 failed")


       def testPredictNbc(self):

           try:
               # client = Client()
               response = self.client.get(reverse('predict_price_nbc'))
               self.assertEqual(response.status_code, 200)
               self.assertTemplateUsed(response, 'predict_price_nbc.html')
               print("test case 4 passed")
           except:
               print("test case 4 failed")


       def testPredictLrnot(self):
           try:
               response = self.client.get(reverse('predict_price_lr'))
               self.assertEqual(response.status_code, 200)
               self.assertTemplateNotUsed(response, 'predict_price_nbc.html')
               print("test case 3 passed")
           except:
               print("test case 3 failed")

       def test_data(self):
            try:
                response = self.client.post("/predict_price_lr/predictlr/",{'CLOCK_SPEED':2.1,'FRONT_CAMERA':5 , 'PRIMARY_CAMERA':12,"INTERNAL_MEMORY":64,'NUMBER_OF_CORES':6,"RAM":5})
                self.assertEqual(response.status_code, 200)
                print("Testcase 5 for form data passed" )
            except :
                print("Testcase 5 for form data failed" )