import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def arnold_catmap(image, iterations):
    s = image.shape[1]
    original = image.copy()
    all_images = []
    I = image.copy()

    plt.ion()
    plt.imshow(original)
    plt.axis('off')
    plt.show()
    plt.pause(2.0)

    for _ in range(iterations):
        for channel in range(I.shape[2]):
            transformed = np.zeros_like(original)
            for x in range(s):
                for y in range(s):
                    new_x = (x + y) % s
                    new_y = (x + 2 * y) % s
                    transformed[new_x, new_y] = I[x, y]

            I = transformed
            all_images.append(transformed)

        print(f"Iteration {_ + 1} completed.")

        
        plt.imshow(transformed)
        plt.axis('off')
        plt.show()

        plt.pause(0.5)

        if(original == transformed).all():
            print(f"Image returned to original after {_ + 1} iterations.")
            plt.pause(2.0)
            plt.ioff() 
            return all_images, _ + 1
    
    plt.ioff() 

    return all_images, iterations
    

if __name__ == "__main__":
    img = Image.open("moon.jpg")
    img = img.resize((100, 100))
    img_array = np.array(img)    

    random_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

    iterations = 150
    transformed_image, iter = arnold_catmap(img_array, iterations)

    print("Original Image Shape:", img_array.shape)
    print("Transformed Image Shape:", transformed_image[0].shape)

    # Display the original, intermidiate and transformed images

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_array)
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"After {iter-2} Iterations") 
    plt.imshow(transformed_image[-3])
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"After {iter} Iterations")
    plt.imshow(transformed_image[-1])
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()