from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import urllib.parse as urlparse
from urllib.parse import parse_qs
import time
import pandas as pd
import os, shutil
import string

from nltk.corpus import words
import nltk

nltk.download('words')



## Importing Necessary Modules


transtable = str.maketrans('', '', string.punctuation)


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


# processing giving page and extracting data

def clean_text(text):
    search_title = ""
    for i in text.split(' '):
        if i.lower() in words.words():
            search_title += i + " "
    return search_title


def download_image_main_page(image_url, image_data):
    try:
        filename = image_url.split("/")[-1]
        filename_path = os.path.join("test_images/", filename)
        # Open the url image, set stream to True, this will return the stream content.

        with open(f'{filename_path}', 'wb') as file:

            # write file
            file.write(image_data.screenshot_as_png)

        print('Image sucessfully Downloaded: ', filename)
    except Exception as e:
        print('Image Couldn\'t be retreived   ', e)

    return filename


def create_dataset(given_url):
    chrome_options = Options()
    # chrome_options.add_argument("--headless")
    driver = webdriver.Chrome('chromedriver.exe', options=chrome_options)
    driver.set_window_size(1920, 1080)


    delete_files()

    driver.get(given_url)
    time.sleep(5)

    title_string = driver.find_elements_by_id('productTitle')[0].text

    price = driver.find_elements_by_id('priceblock_ourprice')[0].text

    search_string = clean_text(title_string)

    search_string = search_string.replace(" ", "+")

    image_data = driver.find_elements_by_id('landingImage')[0]
    image_url = image_data.get_attribute('src')

    filename = download_image_main_page(image_url, image_data)

    test = pd.DataFrame([[0, filename, clean_text(title_string), given_url, price]],
                        columns=["posting_id", "image", "title", "url", "price"])

    url = "https://www.amazon.in/s?k="
    s = url + search_string

    driver.get(s)
    time.sleep(5)

    tt = driver.find_elements_by_class_name('s-main-slot')[0]

    data = []
    i = 0
    for item in tt.find_elements_by_tag_name('div'):

        if item.get_attribute('data-component-type') == 's-search-result':
            image_data = item.find_element_by_class_name('s-image')
            image_url = item.find_element_by_class_name('s-image').get_attribute('src')

            filename = download_image_main_page(image_url, image_data)

            product_name = image_data.get_attribute('alt')

            try:
                price = item.find_element_by_class_name('a-price-whole').text

            except:
                price = "NAN"

            for j in item.find_elements_by_tag_name('a'):
                if j.get_attribute('class') == "a-link-normal a-text-normal":
                    product_link = j.get_attribute('href')

            data.append([i + 1, filename, product_name, product_link, price])
            i += 1

    test1 = pd.DataFrame(data, columns=["posting_id", "image", "title", "url", "price"])
    test = test.append(test1)

    test.to_csv("test1.csv", index=None)
    driver.close()


# doing the search and finding the product related to the give product and extracting all the data required


def find_all_product(soup):
    data = []
    items = soup.find_all("div", attrs={"data-component-type": "s-search-result"})
    print("item found in the page ", len(items))
    # for i in range(5):#len(items)):##### commented here
    for i in range(len(items)):
        img_link = ""
        temp = items[i].find("img", attrs={"class": "s-image"})

        for t in temp.get("srcset").split(",")[-1].split(' '):
            if "https" in t:
                img_link = t

        if img_link == "":
            print("error")
            break
        product_link = tt = items[i].find("a", attrs={"class": "a-link-normal a-text-normal"}).get("href")
        #         print(product_link)
        product_link = "https://www.amazon.in" + product_link
        print(product_link)
        print(img_link)
        print(product_title_clean_text(temp))

        filename = download_image(img_link)

        try:
            price = items[i].find("span", attrs={"class": "a-price-whole"})
            price = price.text

        except:
            price = "NAN"

        data.append([i + 1, filename, product_title_clean_text(temp), product_link, price])

    return data

