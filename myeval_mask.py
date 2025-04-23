import sys
# sys.path.append("..")  # Add parent directory to sys.path
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import cv2
from .models import build_model
from .main import build_postprocessors
from .datasets import transforms as T
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.patches as patches

import argparse
import os
import copy
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# Visualization functions
def show_mask(mask, ax, random_color=False, alpha=1):
    color = np.random.random(4) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image, alpha=alpha)



def show_box(box, score, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='#00ff00', facecolor=(0,0,0,0), lw=2))
    # Display the score above the box
    ax.text(x0+2, y0-10, f'{score:.2f}', color='white', fontsize=30, 
            bbox=dict(facecolor='#00ff00', edgecolor='none'))
    
def show_box_smaller(box, score, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='#00ff00', facecolor=(0,0,0,0), lw=2))
    # Display the score above the box
    ax.text(x0+4, y0-10, f'{score:.2f}', color='white', fontsize=12, 
            bbox=dict(facecolor='#00ff00', edgecolor='none'))


# Visualization functions
def show_mask_multi(masks, ax, scores, random_color=False, save_path = None):
    for i in range(masks.shape[0]):  # Loop over the masks along the first dimension
        mask = masks[i]  # Get the i-th mask
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([255/255, 255/255, 0/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)  # This will overlay the masks on top of each other
        

def show_box_multi(boxes, ax, scores,save_path = None):
    for i in range(boxes.shape[0]):
        box = boxes[i]
        score = scores[0][i]
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor='none', lw=2))
        # Display the score above the box
        ax.text(x0, y0, f'{score:.2f}', color='white', fontsize=30, 
                bbox=dict(facecolor='green', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.1'))

# Image transformation function
def image_transform(img_input=[], image_path=''):
    if image_path:
        img = Image.open(image_path).convert("RGB")
    else:
        img = img_input
    original_height, original_width = img.size[::-1]
    normalize = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = T.Compose([T.RandomResize([800], max_size=1333), normalize])
    img_transformed = transform(img, target=None)[0].unsqueeze(0)
    return img_transformed, original_height, original_width, img_transformed.shape[-2], img_transformed.shape[-1]

def get_bbox(results, n=3):
    bboxes = []
    scores = []
    for result in results:
        index = torch.argsort(result["scores"], descending=True)[:n]
        bboxes = (result["boxes"][index].cpu().numpy())
        scores.append(result["scores"][index].cpu().numpy())
    return bboxes,scores

class ImageProcessor:
    def __init__(self, args, device, seed=42):
        self.seed = seed
        self.set_seed()
        self.args = args
        self.device = device
        self.sam_model_type = args.model_type
        self.sam_checkpoint_path = args.sam_checkpoint_path
        self.ckpt = args.load
        print(self.ckpt)
        self.checkpoint_pronoun = torch.load(self.args.load, map_location="cpu")
        #查看ckpt的key

        self.model, _, _, _ = build_model(args)
        if not self.args.masks:
        #     self.sam_extractor = SamExtractor(arch=self.sam_model_type, pretrained=self.sam_checkpoint_path, device=self.device)
        # else:
            self.sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path).to(self.device)
            self.sam_predictor = SamPredictor(self.sam)
        # Load model state
        load_detr = False
        for key in list(self.checkpoint_pronoun['model_ema'].keys()):
            if 'detr.' in key:
                load_detr = True
                break
        if load_detr and not self.args.masks:
            new_state_dict = {}
            for k,v in self.checkpoint_pronoun['model_ema'].items():
                if k[:5] == 'detr.':
                    name = k[5:]
                    new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict, strict=False)
        else:
            self.model.load_state_dict(self.checkpoint_pronoun['model_ema'], strict=False)
    
    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed) 
        np.random.seed(self.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def process_image_wo_depth(self, phrase, img, req_mask=False, mask=False):
        self.set_seed()
        # Transform image and move to device
        self.img, original_height, original_width, height, width = image_transform(img, '')
        self.img = self.img.to(self.device)
        self.model = self.model.to(self.device)
        
        # Get memory cache from model
        memory_cache = self.model(self.img, phrase, encode_and_save=True)
        
        # Get outputs from model
        outputs = self.model(self.img, phrase, encode_and_save=False, memory_cache=memory_cache)
        
        # Postprocess results
        postprocessors = build_postprocessors(self.args, 'randomly_test')
        orig_size = torch.tensor([original_height, original_width], dtype=torch.int64).unsqueeze(0).to(self.device)
        target_sizes = torch.tensor([height, width], dtype=torch.int64).unsqueeze(0).to(self.device)
        
        results = postprocessors["bbox"](outputs, orig_size)
        boxes, scores = get_bbox(results, n=20)

        filtered_boxes = []
        filtered_scores = []

        for i in range(len(boxes)):
            keep = True
            for j in range(len(filtered_boxes)):
                if self.compute_iou(boxes[i], filtered_boxes[j]) > 0.3: # If IoU is greater than 0.3, keep the box with the higher score
                    if scores[0][i] > filtered_scores[j]:
                        filtered_boxes[j] = boxes[i]
                        filtered_scores[j] = scores[0][i]
                    keep = False
                    break
            if keep:
                filtered_boxes.append(boxes[i])
                filtered_scores.append(scores[0][i])
        
        boxes = filtered_boxes
        scores = [filtered_scores]

        # Filter bboxes by score if multi_label is enabled
        if self.args.multi_label:
            valid_indices = [i for i, score in enumerate(scores[0]) if score > 0.5]
            boxes = [boxes[i] for i in valid_indices]
            scores = [[scores[0][i] for i in valid_indices]]
        else:
            max_score_index = np.argmax(scores[0])
            boxes = [boxes[max_score_index]]
            scores = [[scores[0][max_score_index]]]
            
        
        if mask and req_mask:
            masks = postprocessors["segm"](results, outputs, orig_size, target_sizes)
            masks_to_show = []
            if self.args.multi_label:
                for idx in valid_indices:
                    masks_to_show.append(masks[0]['masks'][idx])
            else:
                masks_to_show.append(masks[0]['masks'][max_score_index])

            image_processed = self.show_mask_and_box(img, masks_to_show, np.array(boxes), scores)
        elif req_mask and not mask:

            image = np.array(img)
            if image.shape[-1] == 3:  
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image)
            masks_to_show = []
            for bb in boxes:
                center = np.array([[(bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2]])
                label = np.array([1])
                # use sam_predictor to get the mask
                mask_to_show, _, _ = self.sam_predictor.predict(
                    point_coords=center,
                    point_labels=label,
                    box=bb,
                    multimask_output=False,
                )
                masks_to_show.append(mask_to_show)

            image_processed = self.show_mask_and_box(img, masks_to_show, boxes, scores)
        else:
            image_processed = self.show_box_only(img, np.array(boxes))
            masks_to_show = None

        # Save mask and bbox if required
        if self.args.save and self.args.save_mask_bbox_path:
            prompt = phrase[0]
            self.save_mask_and_bbox(img, image_processed, masks_to_show, boxes, prompt, scores[:][0])
        # torch.cuda.empty_cache()
        return img, image_processed, masks_to_show, boxes, scores
    
    def compute_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area != 0 else 0

    def process_image_directly(self, phrase, img, depth, req_mask = False, mask=False):
        self.set_seed()
        # Transform image and move to device
        self.img, original_height, original_width, height, width = image_transform(img, '')
        self.img = self.img.to(self.device)
        self.model = self.model.to(self.device)
        
        # Get memory cache from model
        memory_cache = self.model(self.img, phrase, encode_and_save=True)
        
        # Get outputs from model
        outputs = self.model(self.img, phrase, encode_and_save=False, memory_cache=memory_cache)
        
        # Postprocess results
        postprocessors = build_postprocessors(self.args, 'randomly_test')
        orig_size = torch.tensor([original_height, original_width], dtype=torch.int64).unsqueeze(0).to(self.device)
        target_sizes = torch.tensor([height, width], dtype=torch.int64).unsqueeze(0).to(self.device)
        
        results = postprocessors["bbox"](outputs, orig_size)
        box,scores = get_bbox(results,n=1)

        if mask and req_mask:
            masks = postprocessors["segm"](results, outputs, orig_size, target_sizes)
            masks_maxscore_index = torch.argmax(masks[0]['scores'])
        
            # Set SAM extractor size and get embedding
            self.sam_extractor.set_size((original_height, original_width))
            image_embed, interm_embed = self.sam_extractor.get_embedding(self.img)
            self.sam_extractor.set_embedding(image_embed, interm_embed)
            
            # Get coarse mask tensor
            coarse_mask_tensor = masks[0]['masks'][masks_maxscore_index].unsqueeze(0).float()
            refined_mask = self.process_mask_original(depth, coarse_mask_tensor.numpy(), b=0.80, threshold=0.2)
            mask_input = np.expand_dims(refined_mask, axis=0)
            mask_input = np.expand_dims(mask_input, axis=0)
            
            # Sample points from the mask
            points, label = self.sampling(refined_mask, 45, mode='random')
            
            # Refine mask using SAM extractor
            refined_mask = self.sam_extractor.extract_mask_with_given_points(points, label, torch.from_numpy(mask_input), 3).squeeze().cpu().numpy()
            refined_mask = self.process_mask_original(depth, refined_mask, b=0.80, threshold=0.15)
            # Visualize the refined mask
            image_processed = self.show_mask_and_box(img, [refined_mask], np.array(box))
        elif req_mask and not mask:
            #calculate the center of the box
            image = np.array(img)
            if image.shape[-1] == 3:  
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam_predictor.set_image(image)
            box = box[0]
            center = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]])
            label = np.array([1])
            #use sam_predictor to get the mask
            masks_to_show, _, _ = self.sam_predictor.predict(
            point_coords=center,
            point_labels=label,
            box=box,
            multimask_output=False,
            )

            image_processed = self.show_mask_and_box(img, [masks_to_show], np.array([box]), scores)
        else:
            image_processed = self.show_box_only(img, np.array(box))
            masks_to_show = None

        return img, image_processed, masks_to_show, box, scores

    def save_mask_and_bbox(self, img, img_processed, masks, boxes, prompt, score):
        def get_available_prompt_path(base_path, prompt):
            prompt_no_index_path = os.path.join(base_path, prompt, str(0))
            if not os.path.exists(prompt_no_index_path):
                return prompt_no_index_path
            i = 1
            while True:
                candidate = os.path.join(base_path, prompt, str(i))
                if not os.path.exists(candidate):
                    return candidate
                i += 1
        os.makedirs(self.args.save_mask_bbox_path, exist_ok=True)
        # Make a safe filename
        filename = prompt.replace(' ', '_').replace('/', '_')
        prompt_path = get_available_prompt_path(self.args.save_mask_bbox_path, prompt)
        os.makedirs(prompt_path, exist_ok=True)
        # prompt_path = os.path.join(self.args.save_mask_bbox_path, prompt)
        raw_path = os.path.join(prompt_path, 'raw')
        # raw_path = os.path.join(self.args.save_mask_bbox_path, prompt, 'raw')
        os.makedirs(raw_path, exist_ok=True)
        save_path_mask = os.path.join(raw_path, f"{filename}_mask.pdf")
        save_path_bbox = os.path.join(raw_path, f"{filename}_bb.pdf")
        save_path_outline = os.path.join(raw_path, f"{filename}_outline.pdf")
        save_path_combined = os.path.join(prompt_path, f"{filename}_combined.pdf")
        save_path_combined_png = os.path.join(prompt_path, f"{filename}_combined.png")

        # Save mask as black and white vector graphic (PDF)
        if masks is not None:
            if isinstance(masks, list):
                mask_all = np.zeros_like(masks[0].cpu().numpy() if isinstance(masks[0], torch.Tensor) else masks[0])
                for mask in masks:
                    if isinstance(mask, torch.Tensor):
                        mask = mask.cpu().numpy()
                    mask_all = np.maximum(mask_all, np.squeeze(mask))
                    mask_all = np.squeeze(mask_all)
            else:
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                mask_all = np.squeeze(masks.astype(np.uint8))

            fig, ax = plt.subplots()
            ax.imshow(np.zeros_like(img), cmap='gray')  # Black background
            ax.imshow(mask_all, cmap='gray', alpha=1.0, interpolation='none')
            ax.axis('off')
            plt.savefig(save_path_mask, format='pdf', bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            # Save mask outline as black and white vector graphic (PDF)
            self.save_mask_outline_vector(masks, img, save_path_outline)

        # Save bbox image as black and white vector graphic (PDF)
        fig, ax = plt.subplots()
        ax.imshow(np.zeros_like(img), cmap='gray')  # Black background
        for b in boxes:
            x0, y0, x1, y1 = b
            rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)
        ax.axis('off')
        plt.savefig(save_path_bbox, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Save combined image (mask, bbox, outline, original image)
        fig, ax = plt.subplots()
        ax.imshow(img)  # Show the original image

        # Draw masks (blue with 50% transparency)
        if masks is not None:
            for mask in masks:
                show_mask(mask, ax, random_color=False, alpha=0.7)

        # Draw mask outlines (blue, no transparency)
        if isinstance(masks, list):
            for mask in masks:
                if isinstance(mask, torch.Tensor):
                    mask = mask.cpu().numpy()
                mask = np.squeeze(mask)
                self._draw_largest_contours(mask, ax)
        else:
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            masks = np.squeeze(masks)
            if masks is not None:
                if isinstance(masks, list):
                    for mk in masks:
                        if mk is None:
                            continue
                        self._draw_largest_contours(mk, ax)
                else:
                    self._draw_largest_contours(masks, ax)

        # Draw bounding boxes (green, with score)
        for i, box in enumerate(boxes):
            score_to_display = score[i] if isinstance(score, list) else score
            show_box_smaller(box, score_to_display, ax)

        ax.axis('off')
        plt.savefig(save_path_combined, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.savefig(save_path_combined_png, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)


    def save_mask_outline_vector(self, mask, img, save_path_outline):
        fig, ax = plt.subplots()
        ax.imshow(np.zeros_like(img), cmap='gray')  # Black background
        if isinstance(mask, list):
            for m in mask:
                if isinstance(m, torch.Tensor):
                    m = m.cpu().numpy()
                m = np.squeeze(m)
                self._draw_largest_contours(m, ax)
        else:
            if isinstance(mask, torch.Tensor):
                mask = mask.cpu().numpy()
            mask = np.squeeze(mask)
            self._draw_largest_contours(mask, ax)
        ax.axis('off')
        plt.savefig(save_path_outline, format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def _draw_largest_contours(self, mask, ax):
        if mask is None or mask.size == 0:
            return
        # Define color as RGBA (blue color with transparency)
        color = np.array([30/255, 144/255, 255/255, 0.6])
        
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and keep the largest two
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
        
        # Draw each contour
        for contour in contours:
            if cv2.contourArea(contour) > 0:  # Only draw non-zero area contours
                # To plot the contour using the RGBA color, we will plot the line segments explicitly
                xs = contour[:, 0, 0]
                ys = contour[:, 0, 1]
                ax.plot(xs, ys, color=(color[0], color[1], color[2], color[3]), linewidth=2)





    def show_save(self, img, masks, input_box, scores):
        dpi = 100
        w_o, h_o = img.size
        fig_width = w_o / dpi
        fig_height = h_o / dpi

        # plt.figure(figsize=(fig_width, fig_height), facecolor='black')
        plt.figure(figsize=(fig_width, fig_height))
        plt.axis('off')
        plt.tight_layout()  

        plt.imshow(img)

        show_box_multi(masks, plt.gca(), scores)

        show_mask_multi(input_box, plt.gca(), scores)

        # show_points(input_point, input_label, plt.gca())
        # plt.title(f"{phrase[0]}, Score: {max_score:.3f}", fontsize=18)
        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        pixel_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        processed_image = Image.fromarray(pixel_array)
        return processed_image

    def sampling(self, mask, number=35, mode='random'):
        ones_indices = np.argwhere(mask == 1)
        total_points = min(number, ones_indices.size)
        if mode == 'random':
            chosen_indices = np.random.choice(len(ones_indices), size=total_points, replace=False)
            sampled_coords = ones_indices[chosen_indices]
            sampled_coords = np.flip(sampled_coords, axis=1)
        elif mode == 'uniform':
            sampled_coords = np.empty((total_points, 2), dtype=int)
            step = len(ones_indices) // total_points
            for i in range(total_points):
                sampled_coords[i] = ones_indices[i * step]
                sampled_coords = np.flip(sampled_coords, axis=1)
        labels = np.ones(total_points, dtype=int)
        return sampled_coords, labels

    def show_mask_and_box(self, img, masks, boxes, scores=None):
        img = np.asarray(img)
        
        if masks is not None:
            for mask in masks:
                show_mask(mask, plt.gca())
        
        if boxes is not None:
            for box in boxes:
                img = cv2.rectangle(img.copy(), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

        return img

    def show_box_only(self, img, input_box):
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        for box in input_box:
            show_box(box, None, plt.gca())
        plt.axis('off')
        plt.tight_layout()
        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        pixel_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        processed_image = Image.fromarray(pixel_array)
        return processed_image

    def process_mask_original(self, depth, mask, b, threshold):
        mask = np.squeeze(mask)
        depth = depth.reshape(mask.shape[0], mask.shape[1])
        masked_depth = depth * mask
        d_mean = np.mean(masked_depth[masked_depth > 0])
        min_depth = np.min(masked_depth[masked_depth > 0])
        max_depth = np.max(masked_depth[masked_depth > 0])
        masked_depth[masked_depth > 0] = (masked_depth[masked_depth > 0] - min_depth) / (max_depth - min_depth)
        d_mean = (d_mean - min_depth) / (max_depth - min_depth)
        coefficients = np.zeros_like(depth)
        non_zero_depth = masked_depth > 0
        coefficients[non_zero_depth] = np.abs(masked_depth[non_zero_depth] - d_mean)
        diff_up_down = np.abs(np.diff(coefficients, axis=0, prepend=np.zeros((1, coefficients.shape[1]))))
        diff_left_right = np.abs(np.diff(coefficients, axis=1, prepend=np.zeros((coefficients.shape[0], 1))))
        abrupt_changes = np.maximum(diff_up_down, diff_left_right)
        mask[abrupt_changes > threshold] = 0
        mask[coefficients > b] = 0
        return mask.astype(np.uint8)

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="vit_h", type=str, help="Model type for SAM")
    parser.add_argument("--sam_checkpoint_path", default="models/affordance_reasoning/ckpt/sam_vit_h_4b8939.pth", type=str, help="Path to SAM checkpoint")
    parser.add_argument("--image_path", default="models/affordance_reasoning/app_src/outputoriginal_image.png", type=str, help="Path to input image")
    parser.add_argument("--output_path", default="models/affordance_reasoning/app_src/output", type=str, help="Path to output directory")
    parser.add_argument("--load", default="models/affordance_reasoning/ckpt/1144_np_dis.pth", type=str, help="Path to model checkpoint")
    parser.add_argument("--save", action='store_true', help="Whether to save the mask and bbox images")
    parser.add_argument("--save_mask_bbox_path", type=str, default="", help="Path to save mask and bbox images")
    parser.add_argument("--multi_label", action='store_true', help="If enabled, multiple labels are allowed")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args, device

if __name__ == "__main__":
    args, device = init()
    image_process = ImageProcessor(args,device)
    phrase = ["dine on something "]
    img = Image.open(args.image_path).convert("RGB")
    # Assuming depth is available if needed
    depth = None  # Replace with actual depth data if required
    # Call the processing function
    image_process.process_image_directly(phrase, img, depth, req_mask=True, mask=True)
