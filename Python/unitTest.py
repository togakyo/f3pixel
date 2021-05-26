#__main__.py実行で仮想サーバーが作成された環境でユニットテストを実行すること
#実行結果はhttp://localhost:8080/convertへ出力される

import unittest
import json
from urllib.request import urlopen,Request,HTTPError,URLError
from urllib import parse

class testCurrencyConverter(unittest.TestCase):
    
    def __init__(self, name):
        unittest.TestCase.__init__(self, name)
        
    def setUp(self):
        self.url = "http://localhost:8080/convert"      
        
    def webServerRunning(self):
        rc="404"
        
        print ("*** start unitTest webServerRunning ***")
        
        try:
            u=urlopen(self.url) 
            rc = u.getcode()
        except HTTPError as e:
            print('The server couldn\'t fulfill the request.')
            print('Error code: ', e.code)
        except URLError as e:
            print('We failed to reach a server.')
            print('Reason: ', e.reason)
        finally:            
            self.assertEqual(rc, 200, 'webServer is not running!')

    def amountConverted(self):

        res={"status":"failed"}
        
        values = {'amount' : 1}
                
        print ("*** start unitTest amountConverted ***")
        
        try:
            data = parse.urlencode(values)
            data = data.encode('ascii') # data should be bytes
            req = Request(self.url, data)
            with urlopen(req) as response:
                #print(response.read())
                res=json.loads(response.read().decode('utf-8'))                
        except:
            print("Error")
        finally:                    
            self.assertEqual(res['status'], 'success', 'amount has not been converted')

    def badCurrency(self):

        res={"status":"failed"}
        
        values = {'amount' : 1, "src_currency":"BAD"}
                
        print ("*** start unitTest badCurrency ***")
        
        try:
            data = parse.urlencode(values)
            data = data.encode('ascii') # data should be bytes
            req = Request(self.url, data)
            with urlopen(req) as response:
                #print(response.read())
                res=json.loads(response.read().decode('utf-8'))                
        except:
            print("Error")
        finally:
            self.assertEqual(res['status'], 'failed', 'amount has not been converted, ' + res['message'])

def suite():
    suite = unittest.TestSuite()
    suite.addTest(testCurrencyConverter('webServerRunning'))
    suite.addTest(testCurrencyConverter('amountConverted'))
    suite.addTest(testCurrencyConverter('badCurrency'))   
    return suite

def test():
    unittest.TextTestRunner().run(suite())
    
if __name__ == '__main__':
    test()
