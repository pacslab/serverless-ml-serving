FROM node:10-slim

# Reference: https://nodejs.org/fr/docs/guides/nodejs-docker-webapp/

# Create app directory
WORKDIR /usr/src/app

# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
COPY package*.json ./

# RUN npm install
# If you are building your code for production
RUN npm ci --only=production

# Bundle app source
COPY . .

# expose port 3000
EXPOSE 3000

ENTRYPOINT [ "./entrypoint.sh" ]

CMD ["npm", "start"]