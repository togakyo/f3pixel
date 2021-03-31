var invoices = '{"customer": "BigCo","performances": [{"playID": "hamlet","audience": 55},{"playID": "as-like","audience": 35},{"playID": "othello","audience": 40}]}';
var plays = '{"hamlet":{"name": "Hamlet", "type": "tragedy"},"as-like":{"name": "As You Like It", "type": "comedy"},"othello": {"name": "Othello", "type": "tragedy"}}';

var jsonObject_invoices = JSON.parse(invoices);　　　//HACK: invoices JSON読み込みを想定
var jsonObject_plays = JSON.parse(plays);　　　      //HACK: plays JSON読み込みを想定

//Served を立ち上げる
import createStatementData from "./createStatementData.js";

function usd(eNumber){
    return new Intl.NumberFormat("en-US",{ style: "currency", currency: "USD", minimumIntegerDigits: 2 }).format(eNumber/100);
}
function statement (invoices, plays){
    return renderPlainText(createStatementData(invoices, plays));
}

function renderPlainText(data, plays){
    let result =  " Statement for "+ data.customer + "\n"; //'Statement for ${invoices.customer}¥n';
    for (let perf of data.performances){
        result += "  "+ perf.play.name+ ": " + usd(perf.amount) + " " + perf.audience + "seats" + "\n" ;
    }

    result += " Amount owed is " + usd(data.totalAmount) + "\n";
    result += " You earned " + data.totalVolumeCredits + " " +  "credits\n";
    return result ;
}

function renderHtml (data){
    let result = `<h1>Statement for ${data.customer}</h1>\n`; 
    result += "<table>\n";
    result+= "<tr><th>play</th><th>seats</th><th>cost</th></tr>";
    for (let perf of data.performances) {    
        result += `<tr><td>${perf.play.name}</td><td>${usd(perf.amount)}</td><td>(${perf.audience} seats)</td></tr>\n`;
    }
    result += "</table>\n";
    result += `<p>Amount owed is  <em>${usd(data.totalAmount)}</em></p>\n`;
    result += `<p>You earned <em>${data.totalVolumeCredits} "credits</em></p>\n`;
    return result ;
}

function htmlStatement(invoices, plays) {
    return renderHtml(createStatementData(invoices, plays))
}

//function usd(eNumber){
//    return new Intl.NumberFormat("en-US",{ style: "currency", currency: "USD", minimumIntegerDigits: 2 }).format(eNumber/100);
//}


let test = statement(jsonObject_invoices, jsonObject_plays);
let test_html = htmlStatement(jsonObject_invoices, jsonObject_plays);
document.getElementById("box").innerHTML = test_html ;//
console.log(test);

//invoices = '/Users/togashikyousuke/Desktop/f3pixel/javascript/invoices.json';
//plays = '/Users/togashikyousuke/Desktop/f3pixel/javascript/plays.json';

//console.log(jsonObject_invoices.performances);
//console.log(jsonObject_plays.hamlet.name);
//console.log('Hello, World')

