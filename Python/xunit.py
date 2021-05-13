class TestCase:
    def __init__(self, name):
        self.name = name
    def run(self):
        method = getattr(self, self.name)
        method()

class WasRun(TestCase):
    #pass
    def __init__(self, name):
        #pass
        self.wasRun = None
        #self.name = name
        super().__init__(name)
    #def run(self):
        #self.testMethod()
    #    method = getattr(self, self.name)
    #    method()
    def testMethod(self):
        #pass
        self.wasRun = 1

class TestCaseTest(TestCase):
    def testRunning(self):
        test = WasRun("testMethod")
        assert(not test.wasRun)
        test.run()
        assert(test.wasRun)

TestCaseTest("testRunning").run()

#test = Wasrun("testMethod")
#print(test.wasRun)
#test.run()
#print(test.wasRun)
