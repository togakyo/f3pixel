class Wasrun:
    #pass
    def __init__(self, name):
        #pass
        self.wasRun = None
    def run(self):
        self.testMethod()
    def testMethod(self):
        #pass
        self.wasRun = 1
    

test = Wasrun("testMethod")
print(test.wasRun)
test.run()
print(test.wasRun)
