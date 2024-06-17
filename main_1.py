from data.data_management import ImageData 
from models.model import Model
from pipelines import Test

#1 - Car model
#2 - Plate model
#3 - Numbers model
#4 - Test model
		
if __name__ == "__main__":
	m_type = 3

	model = Model(description="The first model", model_type=m_type) # загрузка моедли
	#model.SetUp() # передаем конфигурацию, без параметров - дефолтная
	#model.Build()
	model.LoadEntireModel("numbers") #если хотим загрузить имеющуюся модель (закомментировать SetUp)
	model.Compile()

	image_data = ImageData(m_type) # загрузчик данных
	image_data.LoadDir() # путь - метка
	image_data.Divide(params=(0.00, 0.00, 1.00)) # разделение (можно настроить)
	image_data.LoadImages() # теперь картинка - метка

	#train_data = image_data.GetDataset(part="train") # берется с батчем из настроек

	#hist = model.Fit(train_data, epochs=1)
	#print(hist.history.keys())

	test_data = image_data.GetDataset(part="test") # у тестового просто одно внешнее дополнительное измерение
	losses = Test(model, test_data)
	print("losses: ", losses)


	#model.PrintDescription()
	#model.SaveEntireModel("new_test")