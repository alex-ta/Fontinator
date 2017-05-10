import cv2

image = cv2.imread("C:\\Users\\tschonnieh\\Documents\\Studium\\Repos\\IDA Projekt\\images\\arial\\arial_0.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(gray, 100, 200)

cv2.imshow('fenster', edges)
cv2.waitKey(0)