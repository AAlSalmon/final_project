from ZSIC import ZeroShotImageClassification
from PIL import Image

def classify_clothing2(image_path, gender):

    womens_clothes_categories = [
        "Jeans",
        "Jackets and Coats",
        "Pants",
        "Sweaters",
        "Sweatshirts and Hoodies",
        "T-Shirts and Tanks",
        "Blouses and Shirts",
        "Cardigans",
        "Dresses",
        "Graphic T-Shirts",
        "Jackets and Coats",
        "Leggings",
        "Rompers and Jumpsuits",
        "Shorts",
        "Skirts"
    ]

    mens_clothing_categories = [
        "Jeans",
        "Jackets and Vests",
        "Pants",
        "Shirts and Polos",
        "Suiting",
        "Sweaters",
        "Sweatshirts and Hoodies",
        "T-Shirts and Tanks",
        "Shorts"
    ]
    
    category_mapping = { #maps extracted category to a format that is usable by the dataset, as it has (_).
    "Jackets and Vests": "Jackets_Vests",
    "Shirts and Polos": "Shirts_Polos",
    "Sweatshirts and Hoodies": "Sweatshirts_Hoodies",
    "T-Shirts and Tanks": "Tees_Tanks",
    "Jeans": "Denim",
    "Pants": "Pants",
    "Suiting": "Suiting",
    "Sweaters": "Sweaters",
    "Shorts": "Shorts",
    "Leggings": "Leggings",
    "Skirts": "Skirts",
    "Jackets and Coats": "Jackets_Coats",
    "Blouses and Shirts": "Blouses_Shirts",
    "Cardigans": "Cardigans", 
    "Dresses": "Dresses",
    "Rompers and Jumpsuits": "Rompers_Jumpsuits"
}

    zsic = ZeroShotImageClassification() #defines zero shot image classification
    img = Image.open(image_path) #accesses the image 

    clothes_categories = mens_clothing_categories if gender == 'MEN' else womens_clothes_categories #categoriseing 

    categorisation = zsic(image=img, candidate_labels=clothes_categories) #categorises using zero shot image classification
    
    top_category = categorisation['labels'][categorisation['scores'].index(max(categorisation['scores']))] #extracts category
    mapped_category = category_mapping.get(top_category, top_category) #Maps extracted category such that it matches the ones in the dataset, assigns value to mapped_category varaiable

    return gender, mapped_category  # Returns gender and mapped_category