# ModelShip Production Deployment Guide

Complete guide for deploying ModelShip in production with high availability, security, and monitoring.

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- PostgreSQL 15+
- Redis 7+
- 4GB+ RAM
- 20GB+ Storage

### 1. Environment Setup
```bash
# Copy environment template
cp env.production.template .env

# Edit .env with your production values
nano .env
```

### 2. Generate Security Keys
```bash
# Generate secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set in .env file
SECRET_KEY=your-generated-secret-key
```

### 3. Deploy with Docker Compose
```bash
# Build and start all services
docker-compose -f docker-compose.production.yml up -d

# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f modelship-api
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     Nginx       │    │   ModelShip     │    │   PostgreSQL    │
│  Load Balancer  │────│   API Server    │────│   Database      │
│   & SSL Term    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     Redis       │    │    Celery       │
                       │  Cache & Queue  │────│   Workers       │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Prometheus    │    │    Grafana      │
                       │   Monitoring    │────│   Dashboards    │
                       │                 │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Production Configuration

### Essential Environment Variables

```bash
# Core Settings
ENVIRONMENT=production
DEBUG=false
SECRET_KEY=your-super-secure-secret-key-here

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/modelship_prod

# Security
CORS_ORIGINS=https://yourdomain.com,https://api.yourdomain.com

# Performance
WORKERS=4
MAX_CONCURRENT_CLASSIFICATIONS=10
REDIS_URL=redis://localhost:6379/0

# Monitoring
SENTRY_DSN=your-sentry-dsn-here
METRICS_ENABLED=true
```

### File Upload Limits
```bash
MAX_FILE_SIZE=52428800        # 50MB
MAX_FILES_PER_BATCH=100
MAX_TOTAL_UPLOAD_SIZE=524288000  # 500MB
```

### ML Model Configuration
```bash
HUGGINGFACE_CACHE_DIR=./models_cache
IMAGE_MODEL_NAME=microsoft/resnet-50
TEXT_MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest
OBJECT_DETECTION_MODEL=yolov8n.pt
```

## 🗄️ Database Setup

### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE modelship_prod;
CREATE USER modelship WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE modelship_prod TO modelship;

-- Enable required extensions
\c modelship_prod
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
```

### Database Migration
```bash
# Run migrations
docker-compose exec modelship-api python -c "from database import create_tables; create_tables()"

# Or manually with alembic
docker-compose exec modelship-api alembic upgrade head
```

### Backup Strategy
```bash
# Manual backup
docker-compose exec postgres pg_dump -U modelship modelship_prod > backup.sql

# Automated backup (runs daily at 2 AM)
docker-compose --profile backup run db-backup
```

## 🔒 Security Configuration

### SSL/TLS Setup
1. **Let's Encrypt (Recommended)**
   ```bash
   # Install certbot
   sudo apt install certbot python3-certbot-nginx
   
   # Get certificate
   sudo certbot --nginx -d yourdomain.com -d api.yourdomain.com
   ```

2. **Custom Certificates**
   ```bash
   # Place certificates in nginx/ssl/
   mkdir -p nginx/ssl
   cp your-cert.pem nginx/ssl/
   cp your-key.pem nginx/ssl/
   ```

### Firewall Configuration
```bash
# UFW configuration
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

### Rate Limiting
```bash
# Built-in rate limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_BURST=100

# Additional Nginx rate limiting in nginx.conf
limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
```

## 📊 Monitoring & Observability

### Health Checks
- **API Health**: `GET /health`
- **Detailed Health**: `GET /health/detailed`
- **Metrics**: `GET /health/metrics`
- **Readiness**: `GET /health/ready` (Kubernetes)
- **Liveness**: `GET /health/live` (Kubernetes)

### Monitoring Stack
1. **Prometheus** (Metrics Collection)
   - URL: `http://localhost:9090`
   - Scrapes API metrics every 15s

2. **Grafana** (Dashboards)
   - URL: `http://localhost:3001`
   - Default credentials: admin/admin

3. **Loki** (Log Aggregation)
   - URL: `http://localhost:3100`
   - Collects structured logs

### Key Metrics to Monitor
- **API Response Time**: P95 < 2s
- **Error Rate**: < 1%
- **CPU Usage**: < 80%
- **Memory Usage**: < 85%
- **Disk Usage**: < 90%
- **Database Connections**: < 80% of pool
- **ML Processing Queue**: Length < 100

### Alerting Rules
```yaml
# Prometheus alerts
groups:
  - name: modelship
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 5m
        
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        
      - alert: DatabaseDown
        expr: up{job="postgres"} == 0
        for: 1m
```

## 🚀 Performance Optimization

### Application Tuning
```bash
# Worker configuration
WORKERS=4                    # 2 * CPU cores
MAX_CONCURRENT_CLASSIFICATIONS=10
BATCH_PROCESSING_CHUNK_SIZE=50

# Caching
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600
REDIS_URL=redis://localhost:6379/0

# Compression
ENABLE_COMPRESSION=true
```

### Database Optimization
```sql
-- Recommended PostgreSQL settings
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
```

### ML Model Optimization
- **Model Caching**: Models cached for 24 hours
- **Batch Processing**: Process multiple files together
- **GPU Support**: Set `CUDA_VISIBLE_DEVICES` if available
- **Model Quantization**: Use quantized models for faster inference

## 🔄 Deployment Strategies

### Blue-Green Deployment
```bash
# Deploy new version alongside current
docker-compose -f docker-compose.blue.yml up -d

# Switch traffic after validation
# Update load balancer configuration

# Remove old version
docker-compose -f docker-compose.green.yml down
```

### Rolling Updates
```bash
# Update single service
docker-compose -f docker-compose.production.yml up -d --no-deps modelship-api

# Scale workers
docker-compose -f docker-compose.production.yml up -d --scale worker=6
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: modelship-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: modelship-api
  template:
    metadata:
      labels:
        app: modelship-api
    spec:
      containers:
      - name: modelship-api
        image: modelship/api:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENVIRONMENT
          value: "production"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
```

## 🛠️ Maintenance

### Log Management
```bash
# View application logs
docker-compose logs -f modelship-api

# Rotate logs (configured automatically)
# Logs are rotated at 10MB with 5 backups

# Search logs
docker-compose exec modelship-api grep "ERROR" /app/storage/logs/modelship.log
```

### Database Maintenance
```bash
# Vacuum database
docker-compose exec postgres psql -U modelship -d modelship_prod -c "VACUUM ANALYZE;"

# Check database size
docker-compose exec postgres psql -U modelship -d modelship_prod -c "SELECT pg_size_pretty(pg_database_size('modelship_prod'));"

# Monitor connections
docker-compose exec postgres psql -U modelship -d modelship_prod -c "SELECT count(*) FROM pg_stat_activity;"
```

### System Updates
```bash
# Update Docker images
docker-compose -f docker-compose.production.yml pull

# Restart services
docker-compose -f docker-compose.production.yml up -d

# Clean up old images
docker system prune -a
```

## 🚨 Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Reduce worker count
   WORKERS=2
   ```

2. **Database Connection Issues**
   ```bash
   # Check database status
   docker-compose exec postgres pg_isready -U modelship
   
   # Increase connection pool
   DATABASE_POOL_SIZE=30
   ```

3. **ML Model Loading Errors**
   ```bash
   # Check model files
   ls -la *.pt
   
   # Download models manually
   python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/resnet-50')"
   ```

4. **High API Response Times**
   ```bash
   # Check slow queries
   docker-compose exec postgres psql -U modelship -d modelship_prod -c "SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"
   
   # Enable query caching
   ENABLE_CACHING=true
   ```

### Emergency Procedures

1. **System Overload**
   ```bash
   # Scale down immediately
   docker-compose -f docker-compose.production.yml stop worker
   
   # Reduce API workers
   docker-compose -f docker-compose.production.yml up -d --scale modelship-api=1
   ```

2. **Database Emergency**
   ```bash
   # Create immediate backup
   docker-compose exec postgres pg_dump -U modelship modelship_prod > emergency_backup.sql
   
   # Switch to read-only mode
   docker-compose exec postgres psql -U modelship -d modelship_prod -c "ALTER DATABASE modelship_prod SET default_transaction_read_only = on;"
   ```

## 📞 Support & Monitoring

### Automated Monitoring
- **Uptime Monitoring**: External services (Ping, StatusCake)
- **Error Tracking**: Sentry integration
- **Performance Monitoring**: New Relic/DataDog integration
- **Log Alerting**: Elasticsearch + Kibana

### Manual Checks
- Daily: Check error logs and system resources
- Weekly: Review performance metrics and optimize
- Monthly: Database maintenance and security updates
- Quarterly: Disaster recovery testing

## 🔐 Security Checklist

- [ ] SSL/TLS certificates configured
- [ ] Secret keys changed from defaults
- [ ] Database credentials secured
- [ ] Firewall rules configured
- [ ] Regular security updates applied
- [ ] Access logs monitored
- [ ] Rate limiting enabled
- [ ] File upload restrictions in place
- [ ] CORS properly configured
- [ ] Backup strategy tested

## 📈 Scaling Guidelines

### Vertical Scaling (Single Server)
- **4 CPU cores**: 4 API workers, 2 Celery workers
- **8 CPU cores**: 8 API workers, 4 Celery workers
- **16 CPU cores**: 12 API workers, 8 Celery workers

### Horizontal Scaling (Multiple Servers)
- **Load Balancer**: Nginx or HAProxy
- **Database**: PostgreSQL with read replicas
- **Cache**: Redis Cluster
- **File Storage**: NFS or S3-compatible storage
- **Session Store**: Redis or database

---

## 📚 Additional Resources

- [FastAPI Production Deployment](https://fastapi.tiangolo.com/deployment/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [Redis Production Deployment](https://redis.io/topics/admin)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)
- [Kubernetes Production Deployment](https://kubernetes.io/docs/concepts/cluster-administration/)

For additional support, check the logs at `/app/storage/logs/` or contact the development team. 