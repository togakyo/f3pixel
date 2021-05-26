#!/usr/bin/env python
# -*- coding: utf-8 -*-
#実行結果はhttp://localhost:8080/convertへ出力される

import sys
import cherrypy
import random
import string
import json
from datetime import datetime,timedelta
from platform import python_version
from currency_converter import CurrencyConverter
from currency_converter import RateNotFoundError

CURRENCY_FILE = 'https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist-90d.xml'

class currencyConverterServer(object):
	
	def __init__(self,c):
		self.converter=c
		self.converterDate=datetime.now().strftime("%Y-%m-%d");

	@cherrypy.expose
	def index(self):
		out= """<html>
          <head></head>
          <body>
			<h1>This is a pythonic currency converter API using cherryPy, just for fun.</h1>
			<p>PYVERSION: %s</p>
			<p><b><a href="http://localhost:8080/convert?amount=1">TRY THE API</a></b></p>
          </body>
        </html>"""
		return out %(python_version())
		
	@cherrypy.expose
	def convert(self, amount=0, src_currency="EUR", dest_currency="USD", reference_date=None):
		status='success'
		message='amount converted'
		amountConverted=0

		try:
			
			# automatic update every day
			if(datetime.now().strftime("%Y-%m-%d") != self.converterDate ):
				self.updateCurrencyDict()
				
			# manage empty amount
			if not amount.strip():
				raise ValueError('amount parameter cannot be empty')
			
			# manage empty reference_date
			if not reference_date:
				if self.converter._is_valid_currency(dest_currency):
					reference_date=self.converter.bounds[dest_currency].last_date.strftime('%Y-%m-%d')
				else:
					raise ValueError('{0} is not a supported currency'.format(dest_currency))
				
			amountConverted=self.converter.convert(amount, src_currency, dest_currency, datetime.strptime(reference_date, '%Y-%m-%d').date())

		except (RateNotFoundError,ValueError) as e:
			amountConverted=''
			if reference_date is None:
				reference_date=''
				
			status='failed'
			message=str(e)
		finally:
			ret={
				"amount": amountConverted,
				"currency": dest_currency,
				"exchangeCurrencyDate": reference_date,
				"status": status,
				'message': message
			}
			cherrypy.response.headers['Content-Type'] = 'application/json'			
			return json.dumps(ret).encode('utf8')
		
	def updateCurrencyDict(self):
		cherrypy.expose=False
		self.converter = CurrencyConverter(CURRENCY_FILE)
		

def main():	
	c = CurrencyConverter(CURRENCY_FILE)
	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.quickstart(currencyConverterServer(c))


if __name__ == '__main__':
    main()