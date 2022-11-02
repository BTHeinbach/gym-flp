import owncloud
import os

from dotenv import load_dotenv

load_dotenv()

username = os.getenv("OC_USERNAME")
password = os.getenv("OC_PASSWORD")
oc = owncloud.Client('https://uni-siegen.sciebo.de/')
oc.login(username, password)
oc.logout()

#oc.mkdir('testdir')
#dirs = oc.list('',1)
