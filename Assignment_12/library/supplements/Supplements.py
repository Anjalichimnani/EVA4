from imports.imports_eva import *

class Supplements:
    def __init__():
        super().__init__()
        
    def load_data_json_file (root_path, filename):
        with open (root_path + '/Final_Dog_Annotations.json', 'r') as file:
            data = json.load(file)
            
        return data
        
    def parse_json_image_details(url, data):
        all_images = {}
        X = []
        
        for key in data:
            image = cv2.imread(url + "/{}".format(data[key]["filename"]))
            h, w, c = image.shape
            
            for region in data[key]["regions"]:
                x = region["shape_attributes"]["x"]
                y = region["shape_attributes"]["y"]
                width = region["shape_attributes"]["width"]
                height = region["shape_attributes"]["height"]
                all_images[data[key]["filename"]] = [data[key]["filename"], h, w, x, y, height, width, height/h, width/w]
                X.append([height/h, width/w])
                
        X = np.asarray(X)
        
        return all_images, X
        
    def kmeans_clusters_wcss (X, seed_range, init, max_iter, n_init, random_state):
        wcss = []
        
        for i in range(1, seed_range + 1):
            kmeans = KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
		
        return wcss
        
    def create_anchor_boxes(clusters, image, color, thickness):
        clusters = (clusters) * 500
        
        start_points = (250 - clusters/2).astype(int)
        
        end_points = (250 + clusters/2).astype(int)
        
        for x in range(5):
            start_point = (int(start_points[x][0]), int(start_points[x][1]))
            end_point =  (int(end_points[x][0]), int(end_points[x][1]))
            
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
        
        return image
        
    def show_training_schedule(lr_min, lr_max, train_set_size, batch_size, factor_step_iteration, number_of_cycles):
        lr = []
        iteration_plot = []

        iterations = math.ceil(train_set_size/batch_size)
        stepsize = factor_step_iteration * iterations
        total_iterations = number_of_cycles * iterations

        for i in range(total_iterations):
            cycle = np.floor(1 + i/(2 * stepsize))
            x = np.abs(i/stepsize - 2*cycle + 1)
            lr_value = lr_min + (lr_max - lr_min) * (1 - x)
            lr.append (lr_value)
            iteration_plot.append(i)

        plt.figure(figsize= (20, 3))
        plt.plot (iteration_plot, lr)
        plt.show()
		
    def get_classes_from_file (root_path, filename):
        classes = []
        url = root_path + filename
        
        with open (url) as file:
           classes = file.read().strip().split('\n')
		   
        return classes
		
    def get_class_desc_from_file (root_path, filename):
        classes_desc = {}
        url = root_path + filename

        file = open(url, "r")

        for line in file:
            line_desc = line.split("\t")
            classes_desc[line_desc[0]] = line_desc[1]

        file.close()

        return classes_desc

    def get_train_data_from_file (root_path, classes):
        train_data = []
        train_labels = []

        url = root_path + 'train'

        for class_name in classes:
            for i in range (500):
                train_data.append(plt.imread( (url + "/{}/images/{}_{}.JPEG".format(class_name, class_name, str(i))), 'RGB'))
                train_labels.append(class_name)
                
        return train_data, train_labels
        
        
    def get_val_data_from_file (root_path):
        val_data = []
        val_labels = []

        url = root_path + 'val'

        file = open(url + "/val_annotations.txt", "r")

        for line in file:
            line_desc = line.strip().split("\t")
            val_data.append (plt.imread(url+"/images/{}".format(line_desc[0]), 'RGB'))
            val_labels.append (line_desc[1])

        file.close()
                
        return val_data, val_labels
        
    def concatenate (data1, data2):
        return data1 + data2
        
    def data_shuffle_and_split(data, data_labels, split_perc):
        i = [j for j in range (len(data))]
        random.shuffle(i)
        
        split_size = slice(0,int(split_perc*len(data)))
        split_end = slice(int(split_perc*len(data)), len(data))

        train_data = [data[j] for j in i[split_size]]
        train_labels = [data_labels[j] for j in i[split_size]]

        val_data = [data[j] for j in i[split_end]]
        val_labels = [data_labels[j] for j in i[split_end]]
        
        return train_data, train_labels, val_data, val_labels
        
    def dataset_shuffle_and_split(dataset, split_perc):
        return torch.utils.data.random_split(dataset, [int(split_perc*(len(dataset))), int((1-split_perc)*(len(dataset)))])