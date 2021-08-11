// 仪表盘
var chartDom = document.getElementById('echart1');
var myChart = echarts.init(chartDom);
var option;
option = {
    tooltip: {
        formatter: '{a} <br/>{b} : {c}%'
    },
    series: [{
        name: 'Pressure',
        type: 'gauge',
        detail: {
            valueAnimation: true,
            fontSize: 14,
            color:"#426ab3",
            formatter: '{value}'
        },
        title: {
            fontSize: 14
        },
        data: [{
            value: Math.round(Math.random() * 100),
            name: '日突发值'
        }]
    }]
};
option && myChart.setOption(option);
// 饼图
var chartDom = document.getElementById('echart2');
var myChart = echarts.init(chartDom);
var option;
option = {
    tooltip: {
        trigger: 'item'
    },
    legend: {
        top: '5%',
        left: 0,
        orient:"vertical",
    },
    series: [
        {
            name: '访问来源',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 10,
                borderColor: '#fff',
                borderWidth: 2
            },
            label: {
                show: false,
                position: 'center'
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: '20',
                    fontWeight: 'bold'
                }
            },
            labelLine: {
                show: false
            },
            data: [
                {value: Math.round(Math.random() * 100), name: '事故灾难'},
                {value: Math.round(Math.random() * 100), name: '自然灾害'},
                {value: Math.round(Math.random() * 100), name: '社会安全事件'},
                {value: Math.round(Math.random() * 100), name: '公共卫生事件'},
            ]
        }
    ]
};
option && myChart.setOption(option);
// 条形图1
var chartDom = document.getElementById('echart3');
var myChart = echarts.init(chartDom);
var option;
var data = [];
for (let i = 0; i < 2; ++i) {
    data.push(Math.round(Math.random() * 20));
}
option = {
    title: {
        text: '公共卫生突发事件',
        subtext: '非真实数据'
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ["7月"]
    },
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    },
    xAxis: {
        type: 'value',
        boundaryGap: [0, 0.01]
    },
    yAxis: {
        type: 'category',
        data: ["公共卫生事件","动物疫情"]
    },
    series: [
        {
            name: '7月',
            type: 'bar',
            itemStyle: {
                color:"#f5750c",
            },
            data: data ,
        }
    ]
};
option && myChart.setOption(option);
// 条形图2
var chartDom = document.getElementById('echart4');
var myChart = echarts.init(chartDom);
var option;
var data = [];
for (let i = 0; i < 7; ++i) {
    data.push(Math.round(Math.random() * 20));
}
option = {
    title: {
        text: '自然灾害突发事件',
        subtext: '非真实数据'
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ["7月"]
    },
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    },
    xAxis: {
        type: 'value',
        boundaryGap: [0, 0.01]
    },
    yAxis: {
        type: 'category',
        data: ["水旱灾害","气象灾害","地震灾害","地质灾害","海洋灾害","生物灾害","森林草原火灾"]
    },
    series: [
        {
            name: '7月',
            type: 'bar',
            itemStyle:{
                color:"#3bb255"
            },
            data: data ,
        }
    ]
};
option && myChart.setOption(option);
// 条形图3
var chartDom = document.getElementById('echart5');
var myChart = echarts.init(chartDom);
var option;
var data = [];
for (let i = 0; i < 6; ++i) {
    data.push(Math.round(Math.random() * 20));
}
option = {
    title: {
        text: '社会安全突发事件',
        subtext: '非真实数据'
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ["7月"]
    },
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    },
    xAxis: {
        type: 'value',
        boundaryGap: [0, 0.01]
    },
    yAxis: {
        type: 'category',
        data: ["群体性事件","金融突发事件","涉外突发事件","影响市场稳定\n的突发事件","恐怖袭击事件","刑事案件"]
    },
    series: [
        {
            name: '7月',
            type: 'bar',
            itemStyle: {
                color:"#dd3035",
            },
            data: data ,
        }
    ]
};
option && myChart.setOption(option);
// 条形图4
var chartDom = document.getElementById('echart6');
var myChart = echarts.init(chartDom);
var option;
var data = [];
for (let i = 0; i < 2; ++i) {
    data.push(Math.round(Math.random() * 20));
}
option = {
    title: {
        text: '事故灾害突发事件',
        subtext: '非真实数据'
    },
    tooltip: {
        trigger: 'axis',
        axisPointer: {
            type: 'shadow'
        }
    },
    legend: {
        data: ["7月"]
    },
    grid: {
        left: '3%',
        right: '4%',
        bottom: '3%',
        containLabel: true
    },
    xAxis: {
        type: 'value',
        boundaryGap: [0, 0.01]
    },
    yAxis: {
        type: 'category',
        data: ["安全事故","环境污染和\n生态破坏事故"]
    },
    series: [
        {
            name: '7月',
            type: 'bar',
            itemStyle: {
                color:"#426ab3",
            },
            data: data ,
        }
    ]
};
option && myChart.setOption(option);