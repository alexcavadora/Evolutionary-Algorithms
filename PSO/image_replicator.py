import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import time
from copy import deepcopy

class ImageReplicator:
    def __init__(self, image_path, output_path="output", canvas_size=(200, 200), 
                 max_strokes=100, section_size=20, scale_factor=0.5):
        """
        Initialize the image replicator.
        
        Args:
            image_path: Path to the target image
            output_path: Directory to save output images
            canvas_size: Size of the canvas
            max_strokes: Maximum number of strokes to apply
            section_size: Size of the sections for local comparison
            scale_factor: Factor to scale down the image for faster processing
        """
        self.image_path = image_path
        self.output_path = output_path
        self.max_strokes = max_strokes
        self.section_size = section_size
        self.scale_factor = scale_factor
        
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Load and preprocess the target image
        self.target_image = self.load_and_preprocess_image(image_path, canvas_size)
        self.canvas_size = self.target_image.shape
        
        # Initialize blank canvas
        self.canvas = np.ones(self.canvas_size, dtype=np.uint8) * 255
        
        # PSO parameters
        self.swarm_size = 75
        self.inertia = 0.5
        self.pa = 0.7  # personal acceleration
        self.ga = 0.9  # global acceleration
        self.max_vnorm = 10
        self.num_iters = 25
        
        # Stroke count
        self.stroke_count = 0
        
    def load_and_preprocess_image(self, image_path, canvas_size):
        """Load, convert to grayscale, and resize the target image."""
        try:
            image = Image.open(image_path)
            
            # Convert to grayscale
            image = image.convert('L')
            
            # Resize image
            if self.scale_factor != 1.0:
                new_size = (int(image.width * self.scale_factor), 
                            int(image.height * self.scale_factor))
                image = image.resize(new_size)
            
            # Convert to numpy array
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            print(f"Error loading image: {e}")
            raise
    
    def calculate_fitness(self, params, section_only=True):
        """
        Calculate fitness score (lower is better) by comparing the canvas with target image
        after adding a stroke with the given parameters.
        
        Args:
            params: [x, y, width, height, rotation, color]
            section_only: Whether to compare only the affected section
        
        Returns:
            Fitness score (lower is better)
        """
        # Extract parameters
        x, y, width, height, rotation, color = params
        
        # Convert to appropriate types and ranges
        x = int(x)
        y = int(y)
        width = max(1, int(width))
        height = max(1, int(height))
        rotation = int(rotation) % 360
        color = max(0, min(255, int(color)))
        
        # Create a temporary canvas copy
        temp_canvas = deepcopy(self.canvas)
        
        # Create a temporary image to draw on
        temp_img = Image.fromarray(temp_canvas)
        draw = ImageDraw.Draw(temp_img)
        
        # Calculate corner points for the rectangle
        rect_points = [(x - width/2, y - height/2), 
                      (x + width/2, y + height/2)]
        
        # Draw the rectangle
        draw.rectangle(rect_points, fill=color)
        
        # Convert back to numpy array
        temp_canvas = np.array(temp_img)
        
        # Calculate the section to compare
        if section_only:
            # Define the section boundaries
            x_min = max(0, x - self.section_size)
            y_min = max(0, y - self.section_size)
            x_max = min(self.canvas_size[1], x + width + self.section_size)
            y_max = min(self.canvas_size[0], y + height + self.section_size)
            
            # Compare the section only
            target_section = self.target_image[y_min:y_max, x_min:x_max]
            canvas_section = temp_canvas[y_min:y_max, x_min:x_max]
            
            # Calculate MSE
            mse = np.mean((target_section - canvas_section) ** 2)
        else:
            # Compare the whole image
            mse = np.mean((self.target_image - temp_canvas) ** 2)
        
        return mse
    
    def apply_stroke(self, params):
        """Apply a stroke to the canvas with the given parameters."""
        # Extract parameters
        x, y, width, height, rotation, color = params
        
        # Convert to appropriate types and ranges
        x = int(x)
        y = int(y)
        width = max(1, int(width))
        height = max(1, int(height))
        rotation = int(rotation) % 360
        color = max(0, min(255, int(color)))
        
        # Create an image from the canvas
        img = Image.fromarray(self.canvas)
        draw = ImageDraw.Draw(img)
        
        # Calculate corner points for the rectangle
        rect_points = [(x - width/2, y - height/2), 
                      (x + width/2, y + height/2)]
        
        # Draw the rectangle
        draw.rectangle(rect_points, fill=color)
        
        # Convert back to numpy array
        self.canvas = np.array(img)
        
        # Save progress every 10 strokes
        if self.stroke_count % 1000 == 0:
            self.save_progress()
        
        self.stroke_count += 1
    
    def save_progress(self):
        """Save the current canvas as an image."""
        output_img = Image.fromarray(self.canvas)
        output_img.save(f"{self.output_path}/progress_{self.stroke_count:04d}.png")
    
    def clip_by_norm(self, x, max_norm):
        """Clip vector by norm."""
        norm = np.linalg.norm(x)
        return x if norm <= max_norm else x * max_norm / norm
    
    def optimize_stroke(self):
        """Use PSO to optimize the next stroke."""
        # Parameter bounds: [x, y, width, height, rotation, color]
        bounds = [
            [0, self.canvas_size[1] - 1],  # x
            [0, self.canvas_size[0] - 1],  # y
            [1, 50],  # width
            [1, 50],  # height
            [0, 359],  # rotation
            [0, 255]  # color
        ]
        bounds = np.array(bounds)
        
        # Initialize particles randomly within bounds
        particles = np.zeros((self.swarm_size, len(bounds)))
        velocities = np.zeros((self.swarm_size, len(bounds)))
        
        for i in range(len(bounds)):
            particles[:, i] = np.random.uniform(bounds[i, 0], bounds[i, 1], self.swarm_size)
            velocities[:, i] = np.random.uniform(-1, 1, self.swarm_size)
        
        # Initialize personal and global bests
        personal_bests = np.copy(particles)
        personal_best_fitness = np.array([self.calculate_fitness(p) for p in particles])
        
        global_best_idx = np.argmin(personal_best_fitness)
        global_best = personal_bests[global_best_idx]
        global_best_fitness = personal_best_fitness[global_best_idx]
        
        # PSO iterations
        for i in range(self.num_iters):
            # Print progress every 5 iterations
            if i % 5 == 0:
                print(f"PSO Iteration {i}/{self.num_iters}, Best fitness: {global_best_fitness:.2f}")
            
            # Evaluate fitness for each particle
            for p_i in range(self.swarm_size):
                fitness = self.calculate_fitness(particles[p_i])
                if fitness < personal_best_fitness[p_i]:
                    personal_bests[p_i] = particles[p_i]
                    personal_best_fitness[p_i] = fitness
            
            # Update global best
            min_idx = np.argmin(personal_best_fitness)
            if personal_best_fitness[min_idx] < global_best_fitness:
                global_best_idx = min_idx
                global_best = personal_bests[global_best_idx]
                global_best_fitness = personal_best_fitness[global_best_idx]
            
            # Calculate velocities and update positions
            for p_i in range(self.swarm_size):
                # Momentum
                m = self.inertia * velocities[p_i]
                
                # Personal and global acceleration
                r1, r2 = np.random.rand(), np.random.rand()
                acc_local = self.pa * r1 * (personal_bests[p_i] - particles[p_i])
                acc_global = self.ga * r2 * (global_best - particles[p_i])
                
                # Update velocity
                velocities[p_i] = m + acc_local + acc_global
                velocities[p_i] = self.clip_by_norm(velocities[p_i], self.max_vnorm)
                
                # Update position
                particles[p_i] = particles[p_i] + velocities[p_i]
                
                # Enforce bounds
                for j in range(len(bounds)):
                    particles[p_i, j] = max(bounds[j, 0], min(bounds[j, 1], particles[p_i, j]))
        
        print(f"Optimized stroke: {global_best}, Fitness: {global_best_fitness:.2f}")
        return global_best
    
    def replicate_image(self):
        """Replicate the target image using optimized strokes."""
        start_time = time.time()
        
        # Save initial state
        self.save_progress()
        
        # Optimize and apply strokes
        for i in range(self.max_strokes):
            print(f"\nOptimizing stroke {i+1}/{self.max_strokes}")
            
            # Optimize the next stroke
            stroke_params = self.optimize_stroke()
            
            # Apply the stroke to the canvas
            self.apply_stroke(stroke_params)
            
            # Calculate and print overall error
            overall_error = np.mean((self.target_image - self.canvas) ** 2)
            print(f"Overall Error after stroke {i+1}: {overall_error:.2f}")
            
            # Save the final result
            if i == self.max_strokes - 1:
                self.save_progress()
        
        elapsed_time = time.time() - start_time
        print(f"Image replication completed in {elapsed_time:.2f} seconds")
        
        # Display final result
        self.display_results()
    
    def display_results(self):
        """Display the target image and the replicated image side by side."""
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.target_image, cmap='gray')
        plt.title('Target Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(self.canvas, cmap='gray')
        plt.title('Replicated Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/final_comparison.png")
        plt.show()

def main():
    # Path to the target image
    image_path = "image_bw.png"  # Use your preprocessed BW image
    
    # Create the image replicator
    replicator = ImageReplicator(
        image_path=image_path,
        output_path="test2",
        max_strokes=10000,  # Number of strokes to apply
        section_size=1,  # Size of the sections for local comparison
        scale_factor=0.075  # Scale factor to reduce image size for faster processing
    )
    
    # Replicate the image
    replicator.replicate_image()

if __name__ == "__main__":
    main()