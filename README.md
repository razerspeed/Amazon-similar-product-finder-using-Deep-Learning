# Amazon_similar_product_finder_using_Deep_Learning

In this project an web app has been created to find similar looking amazon products.
This can help find the best deal.


The app users can given a product link and similar products can be found.
The app uses deep learning method to compare images of product.

The model contain a CNN model that find embedding for product images and then find the nearest neighbour using KNN .
For training the model the kaggle competition dataset shopee-product-matching was used.

The model is deployed in AWS

Demo video of App: https://youtu.be/16IZpM6QmW8

Link to Web App : http://13.233.201.126:8501/

Note the web app is not working as amazon is blocking the ip address , hence unable to scrape product.
Solution is to use Amazon API to fetch data.

## HOW TO RUN ##
   
1> pip install -r requirment.txt
    to install required libraries
    
2> Run  python download.py 
   to download the pretrained model
    
3> streamlit run app.py








