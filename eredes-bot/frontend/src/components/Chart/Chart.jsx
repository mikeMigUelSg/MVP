// src/components/Chart/Chart.jsx
import React, { useEffect, useRef, useMemo } from 'react';
import Card from '../Card';
import './Chart.styles.css';

const Chart = ({
  data = [],
  type = 'line',
  title,
  xLabel,
  yLabel,
  color = '#2196f3',
  gradient = true,
  animated = true,
  height = 300,
  className = '',
  ...props
}) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  
  const chartData = useMemo(() => {
    if (!data.length) return [];
    
    // Normalize data
    const maxValue = Math.max(...data.map(d => d.value));
    const minValue = Math.min(...data.map(d => d.value));
    const range = maxValue - minValue || 1;
    
    return data.map(d => ({
      ...d,
      normalizedValue: (d.value - minValue) / range
    }));
  }, [data]);

  const drawChart = (ctx, canvas, progress = 1) => {
    const { width, height } = canvas;
    const padding = 60;
    const chartWidth = width - 2 * padding;
    const chartHeight = height - 2 * padding;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    if (!chartData.length) {
      // Draw empty state
      ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.font = '16px Inter';
      ctx.textAlign = 'center';
      ctx.fillText('Sem dados disponíveis', width / 2, height / 2);
      return;
    }

    // Draw grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
    ctx.lineWidth = 1;
    
    // Vertical grid lines
    for (let i = 0; i <= 6; i++) {
      const x = padding + (chartWidth / 6) * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, padding + chartHeight);
      ctx.stroke();
    }
    
    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding + (chartHeight / 5) * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(padding + chartWidth, y);
      ctx.stroke();
    }

    if (type === 'line') {
      drawLineChart(ctx, chartData, padding, chartWidth, chartHeight, progress);
    } else if (type === 'bar') {
      drawBarChart(ctx, chartData, padding, chartWidth, chartHeight, progress);
    }

    // Draw labels
    drawLabels(ctx, chartData, padding, chartWidth, chartHeight);
  };

  const drawLineChart = (ctx, data, padding, chartWidth, chartHeight, progress) => {
    const pointCount = Math.floor(data.length * progress);
    const visibleData = data.slice(0, pointCount);

    if (visibleData.length === 0) return;

    // Draw gradient fill
    if (gradient) {
      const gradientFill = ctx.createLinearGradient(0, padding, 0, padding + chartHeight);
      gradientFill.addColorStop(0, `${color}40`);
      gradientFill.addColorStop(1, `${color}00`);

      ctx.fillStyle = gradientFill;
      ctx.beginPath();
      ctx.moveTo(padding, padding + chartHeight);
      
      visibleData.forEach((point, index) => {
        const x = padding + (chartWidth / (data.length - 1)) * index;
        const y = padding + chartHeight - (point.normalizedValue * chartHeight);
        
        if (index === 0) {
          ctx.lineTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      
      ctx.lineTo(padding + (chartWidth / (data.length - 1)) * (pointCount - 1), padding + chartHeight);
      ctx.closePath();
      ctx.fill();
    }

    // Draw line
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    
    visibleData.forEach((point, index) => {
      const x = padding + (chartWidth / (data.length - 1)) * index;
      const y = padding + chartHeight - (point.normalizedValue * chartHeight);
      
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();

    // Draw data points
    ctx.fillStyle = color;
    visibleData.forEach((point, index) => {
      const x = padding + (chartWidth / (data.length - 1)) * index;
      const y = padding + chartHeight - (point.normalizedValue * chartHeight);
      
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
      
      // Add glow effect
      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.beginPath();
      ctx.arc(x, y, 2, 0, 2 * Math.PI);
      ctx.fill();
      ctx.shadowBlur = 0;
    });
  };

  const drawBarChart = (ctx, data, padding, chartWidth, chartHeight, progress) => {
    const barWidth = chartWidth / data.length * 0.8;
    const barSpacing = chartWidth / data.length * 0.2;
    
    data.forEach((point, index) => {
      const x = padding + index * (chartWidth / data.length) + barSpacing / 2;
      const barHeight = point.normalizedValue * chartHeight * progress;
      const y = padding + chartHeight - barHeight;
      
      // Create gradient for bar
      const gradient = ctx.createLinearGradient(0, y, 0, y + barHeight);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, `${color}80`);
      
      ctx.fillStyle = gradient;
      ctx.fillRect(x, y, barWidth, barHeight);
      
      // Add border
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, barWidth, barHeight);
    });
  };

  const drawLabels = (ctx, data, padding, chartWidth, chartHeight) => {
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';
    
    // X-axis labels
    const labelCount = Math.min(6, data.length);
    for (let i = 0; i < labelCount; i++) {
      const dataIndex = Math.floor((data.length - 1) * (i / (labelCount - 1)));
      const x = padding + (chartWidth / (labelCount - 1)) * i;
      const label = data[dataIndex]?.label || `${dataIndex}`;
      ctx.fillText(label, x, padding + chartHeight + 20);
    }
    
    // Y-axis labels
    ctx.textAlign = 'right';
    const maxValue = Math.max(...data.map(d => d.value));
    for (let i = 0; i <= 5; i++) {
      const y = padding + chartHeight - (chartHeight / 5) * i;
      const value = (maxValue / 5) * i;
      ctx.fillText(value.toFixed(1), padding - 10, y + 4);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    // Set canvas size for retina displays
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);
    
    if (animated) {
      let startTime = null;
      const duration = 1500;
      
      const animate = (timestamp) => {
        if (!startTime) startTime = timestamp;
        const progress = Math.min((timestamp - startTime) / duration, 1);
        
        // Easing function
        const easeOut = 1 - Math.pow(1 - progress, 3);
        
        drawChart(ctx, canvas, easeOut);
        
        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(animate);
        }
      };
      
      animationFrameRef.current = requestAnimationFrame(animate);
    } else {
      drawChart(ctx, canvas, 1);
    }
    
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [chartData, type, color, gradient, animated]);

  return (
    <Card glass padding="lg" className={`chart-container ${className}`} {...props}>
      {title && (
        <div className="chart-header">
          <h3 className="chart-title">{title}</h3>
        </div>
      )}
      
      <div className="chart-wrapper" style={{ height: `${height}px` }}>
        <canvas
          ref={canvasRef}
          className="chart-canvas"
          style={{ width: '100%', height: '100%' }}
        />
      </div>
      
      {(xLabel || yLabel) && (
        <div className="chart-labels">
          {yLabel && <span className="y-label">{yLabel}</span>}
          {xLabel && <span className="x-label">{xLabel}</span>}
        </div>
      )}
    </Card>
  );
};

export default Chart;