
# coding: utf-8

# ### Imports Libraries

# In[1]:


# !pip install selenium
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import os
import pandas as pd
import numpy as np
import re
import csv
import sys
from time import sleep
from config import *


# In[2]:


try:
    current_path = os.path.dirname(os.path.abspath(__file__))
except:
    current_path = '.'


# In[3]:


def init_driver(gecko_driver='', user_agent='', load_images=True, is_headless=False):
    '''
        This function is just to set up some of default for browser
    '''
    firefox_profile = webdriver.FirefoxProfile()
    
    firefox_profile.set_preference('dom.ipc.plugins.enabled.libflashplayer.so', False)
    firefox_profile.set_preference("media.volume_scale", "0.0")
    firefox_profile.set_preference("dom.webnotifications.enabled", False)
    if user_agent != '':
        firefox_profile.set_preference("general.useragent.override", user_agent)
    if not load_images:
        firefox_profile.set_preference('permissions.default.image', 2)

    options = Options()
    options.headless = is_headless
    
    driver = webdriver.Firefox(options=options,
                               executable_path=f'{current_path}/{gecko_driver}',
                               firefox_profile=firefox_profile)
    
    return driver


# In[4]:


def get_url(url, driver):
    '''
    Argument:
        url of any page to get
        driver that was inilized
    return:
        True
    '''
    driver.get(url)
    sleep(.5)
    return True
# run the driver
driver = init_driver(gecko_driver,user_agent=user_agent)


# In[5]:


def get_first_url(driver, random_url):
    
    # get all pargraphs in the main body and avoid others like not in a box
    pargraphs = driver.find_elements_by_css_selector('#mw-content-text .mw-parser-output p')
    flag = 0
    if not len(pargraphs): # check for body without any pargraphs then it will be without any links
        print("We are in an article without any outgoing Wikilinks !!")
        return False
    else:
        for i in pargraphs:
            # try with the first link we reach in the body of
            try:
                url = i.find_elements_by_css_selector('a')[0].get_attribute('href')
                check_philosophy = re.findall('Getting_to_Philosophy', url)
            
                if check_philosophy == "Getting_to_Philosophy":
                    print("We are finally reached the Philosophy page is ^^")
                    return False
                elif random_url == url:
                    print("We are in the same url !!")
                    return False
                # Once we get a first url we print then break the loop to start with the new link as first link
                print(url)
                random_url = url
                flag = 1
                break
            except:
                pass # because some pargraph does not have a link so we need keep going to first pargraph contain link
        if flag: # run the first link function again with the new first link 
            get_url(random_url ,driver)
            get_first_url(driver, random_url)
        return False # loop over all pargraph without any going links
    

get_url(random_url, driver)
get_first_url(driver, random_url)
    

