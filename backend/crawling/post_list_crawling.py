from bs4 import BeautifulSoup
import re
from tqdm import tqdm
from post_crawling import get_html, Post_Content_Loader, Post_Content_Loader_Campus
import pandas as pd

def post_content_temp():
    return {
        "Title" : [],
        "Category" : [],
        "Date": [],
        "Content" : [],
        "Url" : []
    }

def list_rule_sw(soup):
    board_list = soup.find("ul",{"class":"board-list-wrap"})
    board_list_items = board_list.find_all("li")
    board_list_items = [item.find("a").get("href") for item in board_list_items if item.find("a")]
    board_list_items = [re.search(r'articleNo=(\d+)', url).group(1) for url in board_list_items]
    return board_list_items

def post_list_crawling_with10(URL, csv_name, list_rule, post_rule):
    post_content = post_content_temp()
    for i in tqdm(range(0, 100), desc= "Crawling"):
        soup = BeautifulSoup(
            get_html(URL.format(i * 10))
        )
        board_list_items = list_rule(soup)
        for item in board_list_items:

            ret = post_rule(item)
            if ret:
                post_content["Title"] += [ret["post_title"]]
                post_content["Category"] += [ret["post_category"]]
                post_content["Url"] += [ret["post_url"]]
                post_content["Date"] += [ret["post_date"]]
                post_content["Content"] += [ret["post_content"]]
    

    print(len(post_content["Title"]))
    df = pd.DataFrame(post_content)
    df.to_csv(csv_name, index=False)


def post_list_crawling(URL, csv_name, list_rule, post_rule):
   
    soup = BeautifulSoup(
        get_html(URL)
    )

    board_list_items = list_rule(soup)

    post_content = post_content_temp()
    for item in tqdm(board_list_items, desc= "Crawling"):
        ret = post_rule(item)
        if ret:
            post_content["Title"] += [ret["post_title"]]
            post_content["Category"] += [ret["post_category"]]
            post_content["Url"] += [ret["post_url"]]
            post_content["Date"] += [ret["post_date"]]
            post_content["Content"] += [ret["post_content"]]


    print(len(post_content["Title"]))

    
    df = pd.DataFrame(post_content)

    df.to_csv(csv_name, index=False)

def post_list_crawling_main():
    URL = "https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=list&&articleLimit=10&article.offset={0}"
    URL_MAPPER = lambda articleNo: "https://www.skku.edu/skku/campus/skk_comm/notice01.do?mode=view&articleNo={0}".format(articleNo)
    get_post_content = Post_Content_Loader_Campus(URL_MAPPER)
    post_list_crawling_with10(URL, 'main_content_list.csv', list_rule_sw, get_post_content)
def post_list_crawling_cse():
    URL = "https://cse.skku.edu/cse/notice.do?mode=list&&articleLimit=1000&article.offset=0"
    URL_MAPPER = lambda articleNo: "https://cse.skku.edu/cse/notice.do?mode=view&articleNo={0}".format(articleNo)
    get_post_content = Post_Content_Loader(URL_MAPPER)
    post_list_crawling(URL, 'cse_content_list.csv', list_rule_sw, get_post_content)

def post_list_crawling_cse_grad():
    URL = "https://cse.skku.edu/cse/notice_grad.do?mode=list&&articleLimit=1000&article.offset=0"
    URL_MAPPER = lambda articleNo: "https://cse.skku.edu/cse/notice_grad.do?mode=view&articleNo={0}".format(articleNo)
    get_post_content = Post_Content_Loader(URL_MAPPER)
    post_list_crawling(URL, 'cse_grad_content_list.csv', list_rule_sw, get_post_content)

def post_list_crawling_sw_grad():
    URL = "https://sw.skku.edu/sw/notice_grad.do?mode=list&&articleLimit=1000&article.offset=0"
    URL_MAPPER = lambda articleNo: "https://sw.skku.edu/sw/notice_grad.do?mode=view&articleNo={0}".format(articleNo)
    get_post_content = Post_Content_Loader(URL_MAPPER)
    post_list_crawling(URL, 'sw_grad_content_list.csv', list_rule_sw, get_post_content)


def post_list_crawling_sw():
    URL = "https://sw.skku.edu/sw/notice.do?mode=list&&articleLimit=1000&article.offset=0"
    URL_MAPPER = lambda articleNo: "https://sw.skku.edu/sw/notice.do?mode=view&articleNo={0}".format(articleNo)
    get_post_content = Post_Content_Loader(URL_MAPPER)
    post_list_crawling(URL, 'sw_content_list.csv', list_rule_sw, get_post_content)




if __name__ == "__main__":
    post_list_crawling_main()
    post_list_crawling_cse()
    post_list_crawling_cse_grad()
    post_list_crawling_sw_grad()
    post_list_crawling_sw()