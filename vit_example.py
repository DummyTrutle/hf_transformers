
import tensorflow as tf
import torch

batch_size = 32
image_size = (224, 224)  

data_path = '/home/carla/Owen/ILSVRC_patched2019/ILSVRC/Data/CLS-LOC'

train_dataset = tf.keras.utils.image_dataset_from_directory(
    '/home/carla/Owen/t/',
    validation_split=0.2,  
    subset="training",
    seed=1234567890,
    image_size=image_size,
    batch_size=batch_size,
)

# print(type(train_dataset))
# for i in train_dataset:
#     print(i)


import matplotlib.pyplot as plt
train_dataset = train_dataset.enumerate(start=0)
for e in train_dataset.as_numpy_iterator():
    # print(tf.shape(e[1][0][0]))
    tensor_image = torch.from_numpy(e[1][0][0])
    print(tf.shape(tensor_image))
    img = tensor_image.numpy().transpose(1,2,0)
    a = tensor_image.to(torch.int32)
    plt.imshow(a)
    # tensor_image = tensor_image.view(tensor_image.shape[0], tensor_image.shape[1], tensor_image.shape[2])
    # plt.imshow(tensor_image)
    break


# train_dataset[i]
# plt.imshow(train_dataset.numpy().permute(1,2,0))

# validation_dataset = tf.keras.utils.image_dataset_from_directory(
#     'path/to/custom_dataset',
#     validation_split=0.2,  
#     subset="validation",
#     seed=1234567890,
#     image_size=image_size,
#     batch_size=batch_size,
# )

def second():
    '''
    别忘了pip install vit-keras
    如果是自定义模型的话，调参除了常规deep learning那些lr，
    batch_size，regularization，dropout，
    在ViT可以试试Number of attention heads，Patch_Size...
    '''
    from vit_keras import vit, utils

    image_size = 224     # 224 * 224 *3
    model = vit.vit_l32(
        image_size=image_size,
        activation='softmax', # 二分类sigmoid，class > 2 用softmax
        pretrained=True,       
        include_top=True,      
        pretrained_top=False,
        classes=2 
    )
    # 其实跟最简单的CNN没啥区别，如果只是单独调用ViT_l32
    # (一种pre_trained ViT名字）。

    loss_fn = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # loss function和adam优化器（优化函数？）

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=['accuracy']  
    )


    epochs = 200
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test loss: {test_loss}, Test accuracy: {test_accuracy}")


    #可视化attention map
    import numpy as np
    import matplotlib.pyplot as plt
    from vit_keras import visualize
    attention_map = visualize.attention_map(model=model, image=image)
    print('Prediction:', classes[
        model.predict(vit.preprocess_inputs(image)[np.newaxis])[0].argmax()]
    )
    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.axis('off')
    ax2.axis('off')
    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(image)
    _ = ax2.imshow(attention_map)



# reference: https://mp.weixin.qq.com/s/i5kq_U9-4nlevRMR3BCeaw