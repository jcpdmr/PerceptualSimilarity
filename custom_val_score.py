import json
import numpy as np

def custom_val_score(d0, d1, e0, e1, order_weight=0.7, magnitude_weight=0.3):
    # Punteggio per ordinamento corretto
    correct_order = (d0 > d1 and e0 > e1) or (d0 < d1 and e0 < e1)
    order_score = 1.0 if correct_order else 0.0
    
    # Punteggio per magnitudine relativa
    true_abs_diff = abs(e0 - e1)
    pred_abs_diff = abs(d0 - d1)
    # Gestione casi speciali
    if true_abs_diff == 0 and pred_abs_diff == 0:
        # Se entrambe le differenze sono 0, è un match perfetto
        magnitude_score = 1.0
    elif true_abs_diff == 0 or pred_abs_diff == 0:
        # Se solo una delle due differenze è 0, è il caso peggiore
        magnitude_score = 0.0
    else:
        # Caso normale
        magnitude_score = min(pred_abs_diff/true_abs_diff, true_abs_diff/pred_abs_diff)
    
    return {
        "score" : order_weight * order_score + magnitude_weight * magnitude_score,
        "order_score": order_score,
        "magnitude_score": magnitude_score,
        "weighted_order_score": order_weight * order_score,
        "weighted_magnitude_score": magnitude_weight * magnitude_score
        }

def process_results(input_path, output_path, order_weight=0.7, magnitude_weight=0.3):
    # Load input JSON
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Process each image and calculate scores
    results = {}
    scores = []
    
    for img_name, values in data.items():
        # Extract required values
        d0 = values['d0_deswapped']
        d1 = values['d1_deswapped']
        e0 = values['e24']
        e1 = values['e55']
        
        # Calculate score
        computed_val = custom_val_score(d0, d1, e0, e1, order_weight=order_weight, magnitude_weight=magnitude_weight)
        scores.append(computed_val['score'])
        
        # Store results for this image
        results[img_name] = {
            'd0': d0,
            'd1': d1,
            'e0': e0,
            'e1': e1,
            'custom_val_score': computed_val['score'],
            'pred_diff': d0 - d1,
            'true_diff': e0 - e1,
            "order_score": computed_val["order_score"],
            "magnitude_score": computed_val["magnitude_score"],
            "weighted_order_score": computed_val["weighted_order_score"],
            "weighted_magnitude_score": computed_val["weighted_magnitude_score"],
            'was_swapped': values['was_swapped']
        }
    
    # Calculate mean score
    mean_score = np.mean(scores)
    
    # Prepare final output
    output = {
        'mean_custom_val_score': float(mean_score),
        'num_images': len(scores),
        'order_weight': order_weight,
        'magnitude_weight': magnitude_weight,
        'results': results
    }
    
    # Save output
    output_path = f"{output_path}/custom_val_scores.json" 
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    
    return output

# Example usage
if __name__ == '__main__':
    input_path = 'checkpoints/vgg_LPIPS_pretrained'
    output = process_results(input_path=f"{input_path}/deswapped_results.json", output_path=input_path)
    print(f"Mean validation score: {output['mean_custom_val_score']:.4f}")