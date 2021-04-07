class PerformanceCalculator {
    constructor(aPerformance, aPlay) {
        this.performance = aPerformance;
        this.play = aPlay;
    }

    get amount(){
        throw new Error("サブクラスの責務");
    }
    
    // 通常特典
    get volumeCredits() {
        return Math.max(this.performance.audience - 30, 0);//30人を超えた分のみ特典となる
        //expected 
    }
}

function createPerformanceCalculator(aPerformance, aPlay){
    switch(aPlay.type) {
        case "tragedy": return new TragedyCalculator(aPerformance, aPlay);
        case "comedy": return new ComedyCalculator(aPerformance, aPlay);
        case "Japanese_tragedy": return new JapaneseTragedyCalculator(aPerformance, aPlay);
        default:
            throw new Error("未知の演劇種類 :: "+ aPlay.type); 
    }
}

class TragedyCalculator extends PerformanceCalculator {
    get amount() {
        let result = 40000;
        if(this.performance.audience > 30){
            result += 1000 * (this.performance.audience -30);
        }
        return result;
    }
}
class ComedyCalculator extends PerformanceCalculator {
    get amount() {
        let result = 30000;

        if(this.performance.audience > 20){
            result += 10000 + 500 * (this.performance.audience -20);//20人を超えた分の追加費用
        }
        result += 300 * this.performance.audience;//20人を超えない場合の通常費用
        return result;
    }
    get volumeCredits() {
        return super.volumeCredits + Math.floor(this.performance.audience / 5);//人数分 / 5はさらに追加特典に加算
    }
}
class JapaneseTragedyCalculator extends PerformanceCalculator {
    get amount() {
        let result = 0;
        
        result += 18 * (this.performance.audience);
        
        return result;
    }
    //get volumeCredits() {
    //    return super.volumeCredits + this.performance.audience;
    //}
}

export default function createStatementData(invoices, plays){
    const result = {} ;
    result.customer = invoices.customer;
    result.performances = invoices.performances.map(enrichPerformance);
    console.log(result.performances);//DEBUG_CODE
    result.totalAmount = totalAmount(result);
    result.totalVolumeCredits = totalVolumeCredits(result);
    return result;

    function enrichPerformance(aPerformance){
        const calculator = createPerformanceCalculator(aPerformance, playFor(aPerformance));
        const result = Object.assign({}, aPerformance);
        result.play = calculator.play;
        result.amount = calculator.amount;
        result.volumeCredits = calculator.volumeCredits;
        return result;
    }

    function playFor(aPerformance){
        return plays[aPerformance.playID];
    }

    //function amountFor(aPerformance) {
    //    return new PerformanceCalculator(aPerformance, playFor(aPerformance)).amount;
    //}

    //function volumeCreditsFor(ePerformance){
    //    let result = 0;
    //    result += Math.max(ePerformance.audience - 30, 0) ;
    //    if ("comedy" === ePerformance.play.type) result += Math.floor(ePerformance.audience / 5) ; 
    //    return result;
    //}

    function totalAmount(data){
        
        console.log(data.performances.reduce((total, p) => total + p.amount, 0));//DEBUG_CODE
        //reduce() メソッドは、配列の各要素に対して (引数で与えられた) 関数を実行して、単一の出力値を生成します。
        //reduce() は最初の要素を飛ばしてインデックス 1 から実行されます。initialValue が指定されていたらインデックス 0 から開始します。
        //p.amount 各***calculatorで計算したamountを出力
        //totalに一時保持

        return data.performances.reduce((total, p) => total + p.amount, 0);
    }
    function totalVolumeCredits(data){
        return data.performances.reduce((total, p) => total + p.volumeCredits, 0);
    }
}


