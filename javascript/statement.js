
var invoices = '{"customer": "BigCo","performances": [{"playID": "hamlet","audience": 55},{"playID": "as-like","audience": 35},{"playID": "othello","audience": 40}]}';
var plays = '{"hamlet":{"name": "Hamlet", "type": "tragedy"},"as-like":{"name": "As You Like It", "type": "comedy"},"othello": {"name": "Othello", "type": "tragedy"}}';

var jsonObject_invoices = JSON.parse(invoices);　　　//HACK: invoices JSON読み込みを想定
var jsonObject_plays = JSON.parse(plays);　　　      //HACK: plays JSON読み込みを想定

function amountFor (perf, play){
    let thisAmount = 0;
    

    switch (play.type){
        case "tragedy":
            thisAmount = 40000;
            if(perf.audience > 30){
                thisAmount += 1000 * (perf.audience -30);
            }
            break;
        case "comedy":
            thisAmount = 30000;
            if(perf.audience > 20){
                thisAmount += 10000 + 500 * (perf.audience -20);
            }
            thisAmount += 300 * perf.audience;
            break;
        default:
            throw new Error("unknown type:" + play.type + "\n");
    }

    return thisAmount;
}

function statement (invoices, plays){
    let totalAmount = 0;
    let volumeCredits = 0;
    let result =  " Statement for "+ invoices.customer + "\n"; //'Statement for ${invoices.customer}¥n';

    const format = new Intl.NumberFormat("en-US",
                           { style: "currency", currency: "USD", 
                             minimumIntegerDigits: 2 }).format;
    for (let perf of invoices.performances) {
        const play = plays[perf.playID];

        let thisAmount = amountFor (perf, play);

        //
        volumeCredits += Math.max(perf.audience - 30, 0);
        //
        if("comedy" === play.type) volumeCredits += Math.floor(perf.audience / 5);
        //
        result += "   "+play.name+":" + format(thisAmount/100) + " " + (perf.audience) + " " + "seats\n";
        totalAmount += thisAmount;
    }
    result += " Amount owed is " + format(totalAmount/100) + "\n";
    result += " You earned " + volumeCredits + " " +  "credits\n";
    return result ;
}



let test = statement(jsonObject_invoices, jsonObject_plays);
console.log(test);

//invoices = '/Users/togashikyousuke/Desktop/f3pixel/javascript/invoices.json';
//plays = '/Users/togashikyousuke/Desktop/f3pixel/javascript/plays.json';

//console.log(jsonObject_invoices.performances);
//console.log(jsonObject_plays.hamlet.name);
//console.log('Hello, World')

