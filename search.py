
import sys
import os
import urllib
import socket
import time
import gzip
import re
import random
import types
import urllib.parse
import urllib.error
import io  ## for Python 3
from bs4 import BeautifulSoup
import re
import urllib
import csv
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


import importlib
from dotenv import load_dotenv, find_dotenv
class AppURLopener(urllib.request.FancyURLopener):
    version = "Mozilla/5.0"

opener = AppURLopener()

importlib.reload(sys)
try:
    load_dotenv(find_dotenv(usecwd=True))
    base_url = os.environ.get('BASE_URL')
    results_per_page = int(os.environ.get('RESULTS_PER_PAGE'))
except:
    print ("ERROR: Make sure you have .env file with proper config")
    sys.exit(1)

user_agents = list()


class SearchResult:
    def __init__(self):
        self.url = ''
        self.title = ''
        self.content = ''

    def getURL(self):
        return self.url

    def setURL(self, url):
        self.url = url

    def getTitle(self):
        return self.title

    def setTitle(self, title):
        self.title = title

    def getContent(self):
        return self.content

    def setContent(self, content):
        self.content = content

    def printIt(self, prefix=''):
        print ('url\t->', self.url, '\n',
            'title\t->', self.title, '\n',
            'content\t->', self.content)
        
    def writeFile(self, filename):
        file = open(filename, 'a')
        try:
            file.write('url:' + self.url + '\n')
            file.write('title:' + self.title + '\n')
            file.write('content:' + self.content + '\n\n')
        except IOError:
            print ('file error:')
        finally:
            file.close()


class GoogleAPI:
    def __init__(self):
        timeout = 40
        socket.setdefaulttimeout(timeout)

    def randomSleep(self):
        sleeptime = random.randint(10, 30)
        time.sleep(sleeptime)

    def extractDomain(self, url):
        """Return string
        extract the domain of a url
        """
        domain = ''
        pattern = re.compile(r'http[s]?://([^/]+)/', re.U | re.M)
        url_match = pattern.search(url)
        if(url_match and url_match.lastindex > 0):
            domain = url_match.group(1)

        return domain

    def extractUrl(self, href):
        """ Return a string
        extract a url from a link
        """
        url = ''
        pattern = re.compile(r'(http[s]?://[^&]+)&', re.U | re.M)
        url_match = pattern.search(href)
        if(url_match and url_match.lastindex > 0):
            url = url_match.group(1)

        return url

    def extractSearchResults(self, html):
        """Return a list
        extract serach results list from downloaded html file
        """
        results = list()
        soup = BeautifulSoup(html, 'html.parser')
        div = soup.find('div', id='main')
        if (div == None):
            div = soup.find('div', id='center_col')
        if (div == None):
            div = soup.find('body')

        if (div != None):
            lis = div.findAll('a')
            if(len(lis) > 0):
                i = 0
                for link in lis:
                    i += 1 
                    if (link == None):
                        continue
                    url = link['href']
                    if url.find(".google") > 6:
                        continue
                        
                    url = self.extractUrl(url)
                    if(url == ''):
                        continue
                    title = link.renderContents().decode("utf-8") 
                    title = re.sub(r'<.+?>', '', title)
                    result = SearchResult()
                    result.setURL(url)
                    result.setTitle(title)
                    span = link.find('div')
                    if (span != None):
                        content = span.renderContents().decode("utf-8") 
                        content = re.sub(r'<.+?>', '', content)
                        result.setContent(content)
                    results.append(result)
        return results

    def search(self, query, lang='en', num=results_per_page):
        """Return a list of lists
        search web
        @param query -> query key words
        @param lang -> language of search results
        @param num -> number of search results to return
        """
        search_results = list()
        query = urllib.parse.quote(query)
        if(num % results_per_page == 0):
            pages = num // results_per_page
        else:
            pages = num // results_per_page + 1
        for p in range(0, pages):
            start = p * results_per_page
            url = '%s/search?hl=%s&num=%d&start=%s&q=%s' % (
                base_url, lang, results_per_page, start, query)
            retry = 3
            print("google url", url)
            while(retry > 0):
                try:
                    request = urllib.request.Request(url)
                    
                    length = len(user_agents)
                    index = random.randint(0, length-1)
                    user_agent = user_agents[index]
                    request.add_header('User-agent', user_agent)
                    request.add_header('connection', 'keep-alive')
                    request.add_header('Accept-Encoding', 'gzip')
                    request.add_header('referer', base_url)
                    response = urllib.request.urlopen(request)
                   
                    html = response.read()
                    if(response.headers.get('content-encoding', None) == 'gzip'):
                        html = gzip.GzipFile(
                            fileobj=io.BytesIO(html)).read()
                    results = self.extractSearchResults(html)
                    print(results)
                    search_results.extend(results)
                    break
                except urllib.error.URLError:
                    print ('url error:')
                    self.randomSleep()
                    retry = retry - 1
                    continue

                except Exception:
                    print ('error:')
                    retry = retry - 1
                    self.randomSleep()
                    continue
        return search_results


def load_user_agent():
    fp = open('./user_agents', 'r')

    line = fp.readline().strip('\n')
    while(line):
        user_agents.append(line)
        line = fp.readline().strip('\n')
    fp.close()

#Use Google API to search the keywords
def crawler(keywords):
    # Load use agent string from file
    load_user_agent()

    api = GoogleAPI()
    ref  = []
    # set expect search results to be crawled
    expect_num = 30
    # if no parameters, read query keywords from file
    results = api.search(keywords, num=expect_num)
    for r in results:
        r.printIt()
        item = {}
        item["url"] = r.url
        item["title"] = r.title
        ref.append(item)
    print(ref)
    return ref

#get the HTML document of the URL
#extract title and content from the HTML document
def getUrlContent(url):
    page = opener.open(url)
    content = page.read().decode("utf-8") 
    soup = BeautifulSoup(content, 'lxml')
    news = soup.findAll('p')
    title = soup.find("title").renderContents()
    result = SearchResult()
    result.setContent(news)
    result.setTitle(title)
    return result

'''
if __name__ == '__main__':
    crawler()
'''