import pandas as pd
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from transformers import SamModel, SamConfig, SamProcessor
import torch
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from PIL import Image



app_ui = ui.page_fluid(
    ui.input_file("file1", "Upload Tile image for sidewalk segmentation", accept=".tif", multiple=False),
    ui.output_plot("mask"),  # Changed from ui.output_table to ui.output_plot based on the context of output
)

def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def parsed_file():
        file_info = input.file1()
        if file_info is None or len(file_info) == 0:
            return None
        return file_info[0]["datapath"]
    
    @output
    @render.plot
    async def mask():
        filepath = parsed_file()
        if filepath is None:
            return
        print(filepath)
        # Assuming the model and processor are correctly configured
        model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        my_sidewalk_model = SamModel(model_config)
        my_sidewalk_model.load_state_dict(torch.load("./sidwalk_model_checkpoint.pth", map_location='cpu'))
        device = torch.device("cpu")
        my_sidewalk_model.to(device)

        # Load image
        image = Image.open(filepath)
        imarray = np.array(image)
        single_patch = Image.fromarray(imarray)

        inputs = processor(single_patch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        my_sidewalk_model.eval()
        # Model inference
        with torch.no_grad():
            outputs = my_sidewalk_model(**inputs, multimask_output=False)
        single_patch_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
        # convert soft mask to hard mask
        single_patch_prob = single_patch_prob.cpu().numpy().squeeze()
        single_patch_prediction = (single_patch_prob > 0).astype(np.uint8)


        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the first image on the left
        axes[0].imshow(np.array(single_patch), cmap='gray')  # Assuming the first image is grayscale
        axes[0].set_title("Image")

        # Plot the second image on the right
        axes[1].imshow(single_patch_prob)  # Assuming the second image is grayscale
        axes[1].set_title("Probability Map")

        # Plot the second image on the right
        axes[2].imshow(single_patch_prediction, cmap='gray')  # Assuming the second image is grayscale
        axes[2].set_title("Prediction")

        # Hide axis ticks and labels
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Display the images side by side
        return fig

    

app = App(app_ui, server)
