--Requirement 1. Discuss how database design and indexing strategy optimize performance--
--Requirement 2.  Describe the technical environment used in your database implementation --
	--Normalization 1N to 3N (ensure every column is dependent on primary key, eliminate transitive dependencies)
	--primary indexes (primary key Order_ID)
	--single column index (Country, Item Type, and Region)
--Requirement 3. Demonstrate the functionality of the queries in the lab environment.
--Requirement 4 will be discussed after the queries)

CREATE TABLE Sales_Records (
	Region VARCHAR(50),
	Country VARCHAR(50),
	Item_Type VARCHAR(20),
	Sales_Channel VARCHAR(30),
	Order_Priority CHAR(5),
	Order_Date DATE,
	Order_ID INT PRIMARY KEY,
	Ship_Date DATE,
	Units_Sold INT,
	Unit_Price DECIMAL(20, 2),
	Unit_Cost DECIMAL(20, 2),
	Total_Revenue DECIMAL(20, 2),
	Total_Cost DECIMAL(20, 2),
	Total_Profit DECIMAL(20, 2)
);

--Importing the data --
COPY Sales_Records
From 'C:\WGU\D597\Task 1\Scenario 2\Sales_Records.csv'
	DELIMITER ','
	CSV HEADER;

--Checking the import --
SELECT * FROM Sales_Records

CREATE TABLE Regions (
	Region_ID SERIAL PRIMARY KEY,
	Region_Name VARCHAR(50) UNIQUE
);

CREATE TABLE Countries (
	Country_ID SERIAL PRIMARY KEY,
	Country_Name VARCHAR(50) UNIQUE
);

CREATE TABLE Item_Types (
	Item_Type_ID SERIAL PRIMARY KEY,
	Item_Type_Name VARCHAR(20) UNIQUE
);


INSERT INTO Regions (Region_Name)
SELECT DISTINCT Region 
FROM Sales_Records;

INSERT INTO Countries (Country_Name)
SELECT DISTINCT country 
FROM Sales_Records;

INSERT INTO Item_Types (Item_Type_Name)
SELECT DISTINCT Item_Type 
FROM Sales_Records;

--Check the import in Regions --
SELECT * FROM Regions

--Add the ID's to Sales Records--
ALTER TABLE Sales_Records
	ADD COLUMN Region_ID INT,
	ADD COLUMN Country_ID INT,
	ADD COLUMN Item_Type_ID INT;
	
--Add the new Foreign Key columns --
UPDATE Sales_Records sr
SET Region_ID = r.Region_ID
FROM Regions r
WHERE sr.Region = r.Region_Name;

UPDATE Sales_Records sr
SET Country_ID = c.Country_ID
FROM Countries c
WHERE sr.Country = c.Country_Name;

UPDATE Sales_Records sr
SET Item_Type_ID = it.Item_Type_ID
FROM Item_Types it
WHERE sr.Item_Type = it.Item_Type_Name;

ALTER TABLE Sales_Records
	ADD CONSTRAINT fk_region FOREIGN KEY (Region_ID) REFERENCES Regions(Region_ID),
	ADD CONSTRAINT fk_country FOREIGN KEY (Country_ID) REFERENCES Countries(Country_ID),
	ADD CONSTRAINT fk_item_type FOREIGN KEY (Item_Type_ID) REFERENCES Item_Types(Item_Type_ID);

--Check Sales_Records to see everything loaded properly--

SELECT * FROM Sales_Records
	
--Drop Redundant Tables--
ALTER TABLE Sales_Records 
	DROP COLUMN Region, 
	DROP COLUMN Country, 
	DROP COLUMN Item_Type;
	
--Requirement 4. Discuss how the queries solve the identified business problem--
	--Business problem: Need a flexible, scalable database, and optimization--
	
--Three business queries --
--1. Which Regions are most and least profitable? --
SELECT Region_id, SUM(Total_Profit) AS Total_Profit 
	FROM Sales_Records 
	GROUP BY Region_id 
	ORDER BY Total_Profit DESC;
	
--2. What are the top selling products? --
SELECT Item_Type_id, SUM(Units_Sold) AS Total_Units_Sold 
FROM Sales_Records 
GROUP BY Item_Type_id
ORDER BY Total_Units_Sold DESC 
LIMIT 10;

--3. Which sales channel produces the most revenue--

SELECT Sales_Channel, SUM(Total_Revenue) AS Total_Revenue
FROM Sales_Records
GROUP BY Sales_Channel
ORDER BY Total_Revenue DESC;
