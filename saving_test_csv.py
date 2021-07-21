

## Importing Necessary Modules

from bs4 import BeautifulSoup
import requests
import ast
import pandas as pd
import os, shutil
import string

from nltk.corpus import words
import nltk
nltk.download('words')


transtable = str.maketrans('', '', string.punctuation)
HEADERS = { Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit\
/537.36 (KHTML, like Gecko) Chrome/56.0.2924.87 Safari/537.36
}


def delete_files():
    folder = 'test_images/'
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

#processing giving page and extracting data

def clean_text(text):

    search_title = ""
    for i in text.split(' '):
        if i.lower() in words.words():
            search_title += i + " "
    return search_title


def download_image(image_url):
    filename = image_url.split("/")[-1]
    filename_path = os.path.join("test_images/", filename)
    # Open the url image, set stream to True, this will return the stream content.
    r = requests.get(image_url, stream=True)

    # Check if the image was retrieved successfully
    if r.status_code == 200:
        # Set decode_content value to True, otherwise the downloaded image file's size will be zero.
        r.raw.decode_content = True

        # Open a local file with wb ( write binary ) permission.
        with open(filename_path, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

        print('Image sucessfully Downloaded: ', filename)
    else:
        print('Image Couldn\'t be retreived')

    return filename


def create_dataset(given_url):

    delete_files()
    for attempt in range(10):
        try:
            print("attempt ",attempt)

            webpage = requests.get(given_url, headers=HEADERS)

            soup = BeautifulSoup(webpage.content, "lxml")
            print(soup)
            title = soup.find("span", attrs={"id": 'productTitle'})

            # Inner NavigableString Object
            title_value = title.string

            # Title as a string value
            title_string = title_value.strip()

            # price
            price = soup.find("span", attrs={"id": "priceblock_ourprice"})
            price = price.text

            # Printing Product Title
            print(title_string)

            s = soup.find("img", attrs={"id": 'landingImage'})

            s.get("data-a-dynamic-image")

            t = ast.literal_eval(s.get("data-a-dynamic-image"))

            image_link = list(t)[0]
            print(image_link)
            url = "https://www.amazon.in/s?k="

            search_string = clean_text(title_string)

            search_string = search_string.replace(" ", "+")

            filename = download_image(image_link)

            test = pd.DataFrame([[0, filename, clean_text(title_string), given_url, price]],
                                columns=["posting_id", "image", "title", "url", "price"])

            s = fetch_page(url + search_string)
            data = find_all_product(s)
            test1 = pd.DataFrame(data, columns=["posting_id", "image", "title", "url", "price"])
            test = test.append(test1)

            test.to_csv("test1.csv", index=None)

        except Exception as e:
            print("Exception error",e)
            continue
        else:
            break

    # webpage = requests.get(given_url, headers=HEADERS)
    #
    # soup = BeautifulSoup(webpage.content, "lxml")
    #
    #
    # title = soup.find("span", attrs={"id":'productTitle'})
    #
    # # Inner NavigableString Object
    # title_value = title.string
    #
    # # Title as a string value
    # title_string = title_value.strip()
    #
    # # price
    # price = soup.find("span", attrs={"id": "priceblock_ourprice"})
    # price = price.text
    #
    # # Printing Product Title
    # print(title_string)
    #
    #
    # s=soup.find("img", attrs={"id":'landingImage'})
    #
    # s.get("data-a-dynamic-image")
    #
    # t=ast.literal_eval(s.get("data-a-dynamic-image"))
    #
    # image_link=list(t)[0]
    # print(image_link)
    # url = "https://www.amazon.in/s?k="
    #
    # search_string = clean_text(title_string)
    #
    # search_string = search_string.replace(" ", "+")
    #
    # filename = download_image(image_link)
    #
    # test = pd.DataFrame([[0, filename, clean_text(title_string), given_url,price]],
    #                     columns=["posting_id", "image", "title", "url","price"])
    #
    # s = fetch_page(url + search_string)
    # data = find_all_product(s)
    # test1 = pd.DataFrame(data, columns=["posting_id", "image", "title", "url","price"])
    # test = test.append(test1)
    #
    # test.to_csv("test1.csv", index=None)


#doing the search and finding the product related to the give product and extracting all the data required


def fetch_page(url):
    URL = url
    webpage = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(webpage.content, "lxml")
    return soup


def next_page_link(soup):
    s=soup.find("li", attrs={"class":'a-last'})
    next_page_url="https://www.amazon.com"+s.find("a").get("href")
    return next_page_url
    


def product_title_clean_text(temp):
    title=temp.get("alt")
#     clean_title= title.translate(transtable)
#     search_title=""
#     for i in title.split(' '):
#         if d.check(i):
#             search_title+=i+" "
#     return search_title
    return title


def find_all_product(soup):
    data=[]
    items=soup.find_all("div", attrs={"data-component-type":"s-search-result"})
    print("item found in the page ",len(items))
    # for i in range(5):#len(items)):##### commented here
    for i in range(len(items)):
        img_link=""
        temp=items[i].find("img",attrs={"class":"s-image"}) 
        
        for t in temp.get("srcset").split(",")[-1].split(' '):
            if "https" in t:
                img_link=t
         
        if img_link=="":
            print("error")
            break
        product_link=tt=items[i].find("a",attrs={"class":"a-link-normal a-text-normal"}).get("href")
#         print(product_link)
        product_link="https://www.amazon.in"+product_link
        print(product_link)
        print(img_link)
        print(product_title_clean_text(temp))
        
        filename=download_image(img_link)

        try:
            price=items[i].find("span",attrs={"class":"a-price-whole"})
            price=price.text

        except:
            price="NAN"
        
        data.append([i+1,filename,product_title_clean_text(temp),product_link,price])

        
    return data







