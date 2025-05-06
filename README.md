
README:
1. Open the recommendme folder in VSCode by dragging it 
2. Pip Install Flask, NumPy, Img, Torch, as well as any other imports that I may have forgotten to mention. 
3. move img_highres to your desktop, as it’s hardcoded that way. And change any hardcodings containing the absolute path to your device, these should be the following lines:
Barebones_model.py: line 96: blank_path = “/Users/abdullahalsalem/Desktop/” ← Make this your own path
Deepfashion_vanilla_att2.py: 
Line 8: eval_partition_file='/Users/abdullahalsalem/Desktop/list_eval_partition_modified.txt' ← make this your own path /list_eval_partition_modified.txt
Line 9: image_root='/Users/abdullahalsalem/Desktop'): ← make this your own path
4. move list_eval_partition_modified.txt to your desktop.
5. run Python flask_app.py 

Now, assuming Img_highres can be sent, then this should function well, however, as img_highres is a 7.7GB folder, it might not, so if that does happen, the website won’t function properly. 
Password to unzip img_highres is: mmlab_DeepFashion_inshop. 

If you can't download it from supporting material submission, then open this link "https://drive.google.com/drive/folders/0B7EVK8r0v71pQ2FuZ0k0QnhBQnc?resourcekey=0-NWldFxSChFuCpK4nzAIGsg", go to "In-Shop Clothes Retrieval Benchmark" directory, go to "Img" directory, and download the "img_highres.zip" file, and unzip it using the aforementioned password. 

Note, there should've been a "train_model_save.pth" file, which doesn't upload since its too large to be uploaded onto github. 
