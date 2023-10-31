import requests
import re
from bs4 import BeautifulSoup


def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text
    return _html





class Post_Content_Loader:
    # a function that returns a url with artileNo
    URL_Mapper = None
    def __init__(self, URL_Mapper):
        self.URL_Mapper = URL_Mapper

    def extract_content(self):
        soup_content = ""
        try:
            soup_content = self.soup.find('div', {'class': 'fr-view'}).text.strip()
        except AttributeError:
            soup_content = self.soup.find('div', {'class':'board-view-content-wrap board-view-txt'}).text.strip()
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')  
        soup_content = url_pattern.sub('', soup_content)  
        soup_content = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9 ]', '', soup_content)

    def extract_attributes(self):
        self.attributes = self.soup.find('div', {'class': 'board-view-title-wrap'})


    def load_html_content(self, articleNo):
        self.url = self.URL_Mapper(articleNo)
        html_content = get_html(self.url)
        self.soup = BeautifulSoup(html_content, 'html.parser')
        self.extract_attributes()

    def extract_category(self):
        self.post_category = self.attributes.find('span').text.strip()
        if not (self.post_category.startswith('[') and self.post_category.endswith(']')):
            self.post_category = "[기타]"
        return self.post_category
    
    def extract_date(self):
        return self.attributes.find_all('li')[2].text.strip()
    
    def extract_title(self):
        return self.attributes.find('h4').text[len(self.post_category) + 1:].strip()

    def __call__(self, articleNo):
        soup_content = ""
        try:
            self.load_html_content(articleNo)
            post_category = self.extract_category()   
            post_date = self.extract_date()
            post_title = self.extract_title()
            soup_content = self.extract_content()
            content_summary = soup_content
        except AttributeError as e:
            print(e)
            print("Article {0} Format error".format(articleNo))
            print(soup_content)
            return None
        #print("Post Category : ", post_category)
        #print("Post Title : ", post_title)
        #print("Post Content : ", soup_content)
        #print("Post Summary : ", content_summary)


        return {
            'post_title': post_title,
            'post_category': post_category,
            'post_content': soup_content,
            'post_date': post_date,
            'post_url' : self.url
        }



class Post_Content_Loader_Campus(Post_Content_Loader):
    def extract_content(self):
        soup_content = self.soup.find('dl', {'class': 'board-write-box board-write-box-v03'}).text.strip()
        return soup_content
    
    def extract_attributes(self):
        self.attributes = self.soup.find('table', {'class': 'board_view'}).find('th')

    def extract_date(self):
        return self.attributes.find('span', {'class':'date'}).text.strip().split(':')[1]
    
    def extract_title(self):
        return self.attributes.find('em').text.strip()
    
    def extract_category(self):
        return self.attributes.find('span', {'class':'category'}).text.strip()
