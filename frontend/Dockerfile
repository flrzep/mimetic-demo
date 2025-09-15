# Frontend (CRA) dev Dockerfile
# FROM node:18-alpine
# WORKDIR /app

# # Install deps first to leverage cache
# COPY package.json package-lock.json* .npmrc* ./
# RUN npm install --legacy-peer-deps || npm install

# # Copy source (bind mount will override in dev)
# COPY . .

# ENV HOST=0.0.0.0
# EXPOSE 3000
# CMD ["npm", "start"]


# frontend/Dockerfile
FROM node:22-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
ENV HOST=0.0.0.0
EXPOSE 3000
CMD ["npm", "run", "dev"]