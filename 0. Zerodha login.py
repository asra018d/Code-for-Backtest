# -----------------------------------------------------------------------------------#
# 1. Manual Login
# 2. Automatic Login using Selenium
# -----------------------------------------------------------------------------------#

# Import the required Libraries
from kiteconnect import KiteConnect
from pprint import pprint
import logging
from kiteconnect import KiteTicker
import pandas

#  Set your Account Details Here
api_key = "nx656aq6w8crj09e"
secret_key = "lmtrcmhsx5cfj3xwenw4r9nle7zgjahy"

#microsoft visual C++ installed.
#pip install KiteConnect
#pip install Twisted


kite = KiteConnect(api_key=api_key)

def manualLogin():
    global access_token, kite

    print(" Visit URL , Enter Details and get RequestToken :")
    login_url = "https://kite.trade/connect/login?api_key=" + api_key
    print(login_url)

    code = input("Enter RedirectURI token ")

    data = kite.generate_session(code, api_secret=secret_key)
    kite.set_access_token(data["access_token"])

    accessToken = data["access_token"]
    print("Access Token = ", accessToken)
    access_token = accessToken
    kite.set_access_token(access_token)

    with open("zerodha_api_key.txt", 'w') as file:
        file.write(api_key)
    with open("zerodha_access_token.txt", 'w') as file:
        file.write(accessToken)


manualLogin()


