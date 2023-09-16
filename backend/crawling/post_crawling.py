import requests
import re
from bs4 import BeautifulSoup


def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text
    return _html

def get_post_content(articleNo):
    try:
        html_content = get_html("https://sw.skku.edu/sw/notice.do?mode=view&articleNo={0}".format(articleNo))
        soup = BeautifulSoup(html_content, 'html.parser')
        soup_title = soup.find('div', {'class': 'board-view-title-wrap'})


        post_category = soup_title.find('span').text.strip()
        if not (post_category.startswith('[') and post_category.endswith(']')):
            post_category = "[기타]"
        
        
        post_date = soup_title.find_all('li')[2].text.strip()
        post_title = soup_title.find('h4').text[len(post_category) + 1:].strip()



        try:

            soup_content = soup.find('div', {'class': 'fr-view'}).text.strip()
        except AttributeError:
            soup_content = soup.find('div', {'class':'board-view-content-wrap board-view-txt'}).text.strip()

            
        url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        
        soup_content = url_pattern.sub('', soup_content)
        
        
        soup_content = re.sub(r'[^\uAC00-\uD7A3a-zA-Z0-9 ]', '', soup_content)
        content_summary = soup_content
    except AttributeError:
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
        'post_url' : "https://sw.skku.edu/sw/notice.do?mode=view&articleNo={0}".format(articleNo)
    }



get_post_content(159632)