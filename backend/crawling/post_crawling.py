import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('punkt')

 # Download the Punkt tokenizer data (only need to do this once)

def summarize_korean_text(korean_text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(korean_text, language='korean')

    # Summarize the text
    summary = ""
    current_length = 0

    for sentence in sentences:
        # Calculate the length of the current sentence (considering Korean characters' length)
        sentence_length = len(sentence.encode('utf-8').decode('utf-8'))

        # If adding the current sentence would exceed 100 characters, break the loop
        if current_length + sentence_length > 100:
            break

        # Otherwise, add the sentence to the summary and update the current length
        summary += sentence
        current_length += sentence_length

    return summary




def get_html(url):
    _html = ""
    resp = requests.get(url)
    if resp.status_code == 200:
        _html = resp.text
    return _html

def get_post_content(articleNo):
    html_content = get_html("https://sw.skku.edu/sw/notice.do?mode=view&articleNo={0}".format(articleNo))
    soup = BeautifulSoup(html_content, 'html.parser')
    soup_title = soup.find('div', {'class': 'board-view-title-wrap'})
    post_category = soup_title.find('span').text.strip()
    post_title = soup_title.find('h4').text[len(post_category) + 1:].strip()
    soup_content = soup.find('div', {'class': 'fr-view'}).text.strip()
    content_summary = summarize_korean_text(soup_content)
    print("Post Category : ", post_category)
    print("Post Title : ", post_title)
    print("Post Content : ", soup_content)
    print("Post Summary : ", content_summary)



get_post_content(159632)