// Пользовательские подпрограммы
let rec factorial n =
    match n with
    | 0 | 1 -> 1
    | _ when n > 0 -> n * factorial (n - 1)
    | _ -> failwith "Отрицательное число"

let calculateCircleStats radius =
    let area = Math.PI * radius * radius
    let circumference = 2.0 * Math.PI * radius
    (area, circumference)

let processList items =
    items
    |> List.filter (fun x -> x % 2 = 0)
    |> List.map (fun x -> x * x)
    |> List.sortDescending

// Функции с несколькими параметрами
let calculateRectangleArea length width =
    length * width

let calculateCompoundInterest principal rate years =
    let mutable amount = principal
    for _ in 1..years do
        amount <- amount * (1.0 + rate / 100.0)
    amount

let findMaxOfThree a b c =
    if a >= b && a >= c then a
    elif b >= a && b >= c then b
    else c

let formatPersonalInfo firstName lastName age city =
    sprintf "Имя: %s %s, Возраст: %d, Город: %s" firstName lastName age city

let calculateDistance x1 y1 x2 y2 =
    let dx = x2 - x1
    let dy = y2 - y1
    sqrt (dx * dx + dy * dy)

// Основная программа
[<EntryPoint>]
let main argv =
    // Основные операторы
    printfn "Введите число для вычисления факториала:"
    let input = Console.ReadLine()
    
    let number = 
        match Int32.TryParse input with
        | (true, n) when n >= 0 -> n
        | _ -> 0

    let factResult = factorial number
    printfn "Факториал %d равен %d" number factResult

    // Работа с кортежами
    let radius = 5.0
    let (area, circumference) = calculateCircleStats radius
    printfn "Площадь круга: %.2f, Длина окружности: %.2f" area circumference

    // Работа со списками
    let numbers = [1; 2; 3; 4; 5; 6; 7; 8; 9; 10]
    let processed = processList numbers
    printfn "Обработанный список: %A" processed

    // Условные операторы
    let message = 
        if number > 5 then "Больше пяти"
        elif number > 0 then "Положительное"
        else "Ноль или отрицательное"
    
    printfn "%s" message

    // Циклы
    printfn "Числа от 1 до 5:"
    for i in 1..5 do
        printf "%d " i
    printfn ""

    printfn "Четные числа до 10:"
    let mutable counter = 2
    while counter <= 10 do
        printf "%d " counter
        counter <- counter + 2
    printfn ""

    // Использование функций с несколькими параметрами
    let rectangleArea = calculateRectangleArea 10 5
    printfn "Площадь прямоугольника: %d" rectangleArea

    let investmentResult = calculateCompoundInterest 1000.0 5.0 10
    printfn "Сумма инвестиций через 10 лет: %.2f" investmentResult

    let maxValue = findMaxOfThree 42 17 99
    printfn "Максимальное из трех чисел: %d" maxValue

    let personInfo = formatPersonalInfo "Иван" "Петров" 35 "Москва"
    printfn "%s" personInfo

    let distance = calculateDistance 0.0 0.0 3.0 4.0
    printfn "Расстояние между точками: %.2f" distance

    // Обработка аргументов командной строки
    match argv with
    | [| |] -> printfn "Аргументы не предоставлены"
    | args ->
        printfn "Получены аргументы:"
        args |> Array.iter (printfn "- %s")

    // Сопоставление с образцом
    let testPattern x =
        match x with
        | 1 -> "Единица"
        | 2 -> "Двойка"
        | _ when x < 0 -> "Отрицательное"
        | _ -> "Другое число"

    printfn "%s" (testPattern number)

    // Запрос ввода
    Console.WriteLine("Нажмите любую клавишу для выхода...")
    Console.ReadKey() |> ignore

    0 // код возврата
