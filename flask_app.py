from flask import Flask, render_template, request, url_for
import os
from categoriser import classify_clothing2
from deepfashion_vanilla_att2 import create_deepfashion_datasets
from barebones_model import TransformedGalleryDataset, load_model, get_embedding, find_similar_items, display_ids_and_distances
from torch.utils.data import DataLoader
import shutil

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        uploaded_image = request.files.get('submit_image_test') #takes input image using request.files
        selected_gender = request.form.get('gender') #takes input gender (MEN/WOMEN)
        
        if uploaded_image and uploaded_image.filename != '': #If the image has been submitted
            save_path = os.path.join('static/uploads', uploaded_image.filename) #creates path in static/uploads + uploaded images path
            uploaded_image.save(save_path) #saves path, and therefore access to the image 
            
            # Get classification and recommendations
            i_gender, clothing_category = classify_clothing2(save_path, selected_gender)
            dataset, entries = create_deepfashion_datasets(i_gender, clothing_category)
            transformed_gallery = TransformedGalleryDataset(dataset)
            gallery_loader = DataLoader(transformed_gallery, batch_size=32, shuffle=False)
            model = load_model()
            query_embed = get_embedding(model, save_path)
            similar_items = find_similar_items(query_embed, gallery_loader, model, entries, top_k=20)
            returned_images = display_ids_and_distances(similar_items)

            rec_folder = 'static/uploads/recommendations' #get folder path to recommendations, where returned images will be placed
            
            for f in os.listdir(rec_folder): # clears the images in the recommendations, before re-populating it with the recommendations based on the current inputted image
                os.remove(os.path.join(rec_folder, f))
            
            # Prepare web-accessible image paths
            web_images = []
            for i, img_path in enumerate(returned_images):
                rec_filename = f"rec_{i}_{os.path.basename(img_path)}" #creates new filenames for entries so it can be accessed by the browser
                dest_path = os.path.join(rec_folder, rec_filename) #joins that new filename to directory, to create a path so that the browser can access it from static/uploads/recommendations
                shutil.copy2(img_path, dest_path) #copies contents (image) from original path onto the new path, which is accessable by the browser
                web_path = url_for('static', filename=f"uploads/recommendations/{rec_filename}") #creates urls for browser to access the returned images // filename=f
                web_images.append(web_path) #appends them to web_images, which will contain the returned images and pass them to the recommendations page
            
            return render_template('recommendation_page.html', #renders page successfully
                                users_image=url_for('static', filename=f"uploads/{uploaded_image.filename}"), #accesses inputted image and returns image to recommendation page to display
                                gender=selected_gender,
                                returned_images=web_images)
        
        return render_template('recommendation_page.html', error="No image uploaded") #if there is not uploaded image, and returns an error 
    
    return render_template('main_page.html')

if __name__ == "__main__":
    app.run(debug=True)